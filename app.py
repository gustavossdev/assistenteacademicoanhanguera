"""Assistente Acadêmico — chatbot com texto e voz (Streamlit + Ollama + Faster-Whisper).

Execute com:  streamlit run app.py

Organização do arquivo (de baixo para cima na dependência):
    1. Configurações        -> Settings
    2. Infraestrutura       -> configuração do FFmpeg
    3. Repository           -> JsonCursoRepository
    4. Prompt               -> PromptBuilder
    5. Strategy/Adapter     -> LLMClient / OllamaClient, SpeechToText / WhisperTranscriber
    6. Facade               -> ChatService
    7. Interface (Streamlit)-> Historico e componentes de UI
    8. Composition root     -> criar_servico() e main()
"""
from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol

import numpy as np
import requests
import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
from pydub import AudioSegment


# 1. CONFIGURAÇÕES E CONSTANTES

BASE_DIR = Path(__file__).resolve().parent

CHAVE_MENSAGENS = "messages"
CHAVE_ULTIMO_AUDIO = "ultimo_audio_hash"

TEXTO_SUBTITULO = (
    "Tire dúvidas sobre cursos, ementas e disciplinas da faculdade "
    "por **texto** ou **voz** (barra lateral)."
)

MSG_ERRO_CONEXAO = "⚠️ Não consegui conectar ao Ollama. Verifique se ele está rodando (`ollama serve`)."
MSG_ERRO_TIMEOUT = "⏳ O modelo demorou para responder (timeout). Tente novamente."
MSG_ERRO_GENERICO = "⚠️ Erro no modelo de linguagem: {detalhe}"

TEMPLATE_PROMPT_SISTEMA = """Você é um assistente acadêmico virtual de uma universidade.
Seu objetivo é ajudar alunos de forma educada, amigável e clara.
Você deve responder baseando-se EXCLUSIVAMENTE na base de conhecimento em formato JSON abaixo.

REGRA DE SEGURANÇA MUITO IMPORTANTE:
Se o aluno perguntar sobre um curso, matéria, valor ou qualquer informação que NÃO exista na base de conhecimento, você NÃO deve inventar respostas.
Nesse caso, responda de forma muito educada e acolhedora que você não possui essa informação ou que a universidade ainda não oferece esse curso/opção no momento.
Exemplo de tom desejado: "Desculpe, mas ainda não oferecemos o curso de Medicina na nossa instituição." ou "Puxa, não tenho a informação sobre mensalidades no momento."

BASE DE CONHECIMENTO:
{base_conhecimento}"""

Curso = Dict[str, Any]
Mensagem = Dict[str, str]


@dataclass(frozen=True)
class Settings:
    """Todas as configurações em um único objeto imutável."""

    arquivo_cursos: Path = BASE_DIR / "dados" / "cursos.json"

    # LLM (Ollama)
    ollama_url: str = "http://localhost:11434/api/generate"
    modelo_llm: str = "llama3.2"
    timeout_llm_s: int = 180

    # Speech-to-Text (Faster-Whisper)
    whisper_modelo: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    idioma: str = "pt"
    taxa_amostragem_hz: int = 16_000

    # Interface
    titulo_pagina: str = "Assistente Acadêmico"
    icone_pagina: str = "🎓"

    @property
    def descricao_whisper(self) -> str:
        return f"{self.whisper_modelo} ({self.whisper_device.upper()}/{self.whisper_compute_type})"


SETTINGS = Settings()


# 2. INFRAESTRUTURA: FFMPEG

def localizar_ffmpeg_winget() -> Optional[Path]:
    """Procura a pasta bin do FFmpeg instalado via winget (sem depender da versão)."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None

    pacotes = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    candidatos = sorted(pacotes.glob("Gyan.FFmpeg*/*/bin"))
    return candidatos[-1] if candidatos else None


def configurar_ffmpeg() -> None:
    """Adiciona o FFmpeg ao PATH e ao pydub."""
    bin_dir = localizar_ffmpeg_winget()
    if bin_dir is None:
        return

    path_atual = os.environ.get("PATH", "")
    if str(bin_dir) not in path_atual.split(os.pathsep):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + path_atual

    executavel = bin_dir / "ffmpeg.exe"
    if executavel.exists():
        AudioSegment.converter = str(executavel)
        AudioSegment.ffmpeg = str(executavel)


configurar_ffmpeg()


# =============================================================================
# 3. REPOSITORY: ACESSO À BASE DE CONHECIMENTO
# =============================================================================
class CursoRepository(Protocol):
    """Contrato para qualquer fonte de cursos (JSON, banco de dados, API...)."""

    def listar(self) -> List[Curso]:
        ...


class JsonCursoRepository:
    """Lê os cursos de um arquivo JSON."""

    def __init__(self, caminho: Path) -> None:
        self._caminho = caminho

    def listar(self) -> List[Curso]:
        try:
            with open(self._caminho, encoding="utf-8") as arquivo:
                conteudo = json.load(arquivo)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

        return conteudo.get("cursos", []) if isinstance(conteudo, dict) else conteudo


# 4. PROMPT

class PromptBuilder:
    """Monta o prompt de sistema uma única vez (a base não muda entre perguntas)."""

    def __init__(self, cursos: List[Curso]) -> None:
        base_json = json.dumps(cursos, ensure_ascii=False, indent=2)
        self._prompt_sistema = TEMPLATE_PROMPT_SISTEMA.format(base_conhecimento=base_json)

    @property
    def prompt_sistema(self) -> str:
        return self._prompt_sistema


# 5. STRATEGY / ADAPTER: LLM E SPEECH-TO-TEXT

class LLMError(Exception):
    """Erro genérico do provedor de LLM."""


class LLMConexaoError(LLMError):
    """Não foi possível conectar ao provedor."""


class LLMTimeoutError(LLMError):
    """O provedor demorou demais para responder."""


class LLMClient(ABC):
    """Contrato para qualquer provedor de LLM (Ollama, OpenAI, etc.)."""

    @abstractmethod
    def gerar_stream(self, pergunta: str, sistema: str) -> Iterator[str]:
        """Gera a resposta token a token. Levanta subclasses de LLMError em falhas."""


class OllamaClient(LLMClient):
    """Adapter para a API HTTP local do Ollama."""

    def __init__(self, settings: Settings) -> None:
        self._url = settings.ollama_url
        self._modelo = settings.modelo_llm
        self._timeout = settings.timeout_llm_s

    def gerar_stream(self, pergunta: str, sistema: str) -> Iterator[str]:
        payload = {
            "model": self._modelo,
            "system": sistema,
            "prompt": pergunta,
            "stream": True,
        }

        try:
            with requests.post(self._url, json=payload, stream=True, timeout=self._timeout) as resp:
                resp.raise_for_status()
                for linha in resp.iter_lines(decode_unicode=True):
                    if not linha:
                        continue
                    dados = json.loads(linha)
                    yield dados.get("response", "")
                    if dados.get("done", False):
                        break
        except requests.exceptions.ConnectionError as erro:
            raise LLMConexaoError(str(erro)) from erro
        except requests.exceptions.Timeout as erro:
            raise LLMTimeoutError(str(erro)) from erro
        except requests.exceptions.HTTPError as erro:
            raise LLMError(f"HTTP {erro.response.status_code}: {erro.response.text}") from erro


class SpeechToText(ABC):
    """Contrato para qualquer motor de transcrição de voz."""

    @abstractmethod
    def transcrever(self, audio: AudioSegment) -> str:
        """Converte áudio em texto. Retorna string vazia se não houver fala."""


class WhisperTranscriber(SpeechToText):
    """Transcrição local usando Faster-Whisper."""

    def __init__(self, settings: Settings) -> None:
        self._idioma = settings.idioma
        self._taxa = settings.taxa_amostragem_hz
        self._modelo = WhisperModel(
            settings.whisper_modelo,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    def transcrever(self, audio: AudioSegment) -> str:
        if audio is None or len(audio) == 0:
            return ""

        amostras = self._para_numpy(audio)
        segmentos, _ = self._modelo.transcribe(amostras, language=self._idioma)
        return " ".join(segmento.text for segmento in segmentos).strip()

    def _para_numpy(self, audio: AudioSegment) -> np.ndarray:
        """Reamostra para mono/16 kHz e normaliza para float32 em [-1.0, 1.0]."""
        audio = audio.set_frame_rate(self._taxa).set_channels(1)
        amostras = np.array(audio.get_array_of_samples(), dtype=np.float32)
        amostras /= float(2 ** (audio.sample_width * 8 - 1))
        return amostras


# 6. FACADE: SERVIÇO DE CHAT

class ChatService:
    """Ponto único de acesso da interface às regras do chatbot.

    As dependências são injetadas (Dependency Injection), o que facilita
    testes e a troca de implementações.
    """

    def __init__(
        self,
        repositorio: CursoRepository,
        llm: LLMClient,
        stt: SpeechToText,
        nome_modelo_llm: str,
        nome_modelo_stt: str,
    ) -> None:
        self._llm = llm
        self._stt = stt
        self._cursos = repositorio.listar()
        self._prompt_sistema = PromptBuilder(self._cursos).prompt_sistema
        self.nome_modelo_llm = nome_modelo_llm
        self.nome_modelo_stt = nome_modelo_stt

    @property
    def total_cursos(self) -> int:
        return len(self._cursos)

    def responder_stream(self, pergunta: str) -> Iterator[str]:
        """Responde em streaming, convertendo falhas técnicas em mensagens amigáveis."""
        try:
            yield from self._llm.gerar_stream(pergunta, self._prompt_sistema)
        except LLMConexaoError:
            yield MSG_ERRO_CONEXAO
        except LLMTimeoutError:
            yield MSG_ERRO_TIMEOUT
        except LLMError as erro:
            yield MSG_ERRO_GENERICO.format(detalhe=erro)

    def transcrever(self, audio: AudioSegment) -> str:
        return self._stt.transcrever(audio)


# 7. INTERFACE (STREAMLIT) — sem regras de negócio

class Historico:
    """Encapsula o histórico de conversa guardado no session_state."""

    def __init__(self) -> None:
        if CHAVE_MENSAGENS not in st.session_state:
            st.session_state[CHAVE_MENSAGENS] = []

    @property
    def mensagens(self) -> List[Mensagem]:
        return st.session_state[CHAVE_MENSAGENS]

    def adicionar(self, papel: str, conteudo: str) -> None:
        self.mensagens.append({"role": papel, "content": conteudo})

    def limpar(self) -> None:
        st.session_state[CHAVE_MENSAGENS] = []


def renderizar_historico(historico: Historico) -> None:
    for mensagem in historico.mensagens:
        with st.chat_message(mensagem["role"]):
            st.markdown(mensagem["content"])


def processar_pergunta(pergunta: str, servico: ChatService, historico: Historico) -> None:
    """Renderiza a pergunta e a resposta em streaming, registrando ambas no histórico."""
    historico.adicionar("user", pergunta)
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        resposta = st.write_stream(servico.responder_stream(pergunta))

    historico.adicionar("assistant", resposta)


def capturar_voz(servico: ChatService) -> Optional[str]:
    """Grava e transcreve a fala; retorna o texto apenas para áudios novos."""
    st.subheader("🎤 Entrada por Voz")
    st.caption("Clique no botão abaixo para gravar sua pergunta.")

    audio = audiorecorder(
        start_prompt="●  Gravar",
        stop_prompt="■  Parar",
        pause_prompt="",
        key="mic",
        show_visualizer=True,
    )
    if len(audio) == 0:
        return None

    assinatura = hashlib.sha1(audio.raw_data).hexdigest()
    if assinatura == st.session_state.get(CHAVE_ULTIMO_AUDIO):
        return None
    st.session_state[CHAVE_ULTIMO_AUDIO] = assinatura

    with st.spinner("Transcrevendo..."):
        texto = servico.transcrever(audio)

    if texto:
        st.success(f"🗣️ *{texto}*")
        return texto

    st.warning("Não entendi o áudio. Tente novamente.")
    return None


def renderizar_sidebar(servico: ChatService, historico: Historico) -> Optional[str]:
    """Desenha a barra lateral e devolve o texto transcrito, se houver."""
    with st.sidebar:
        st.header("⚙️ Painel de Controle")
        st.caption(f"**Modelo LLM:** `{servico.nome_modelo_llm}`")
        st.caption(f"**Whisper:** `{servico.nome_modelo_stt}`")
        st.caption(f"**Cursos catalogados:** {servico.total_cursos}")
        st.divider()

        texto_transcrito = capturar_voz(servico)

        st.divider()
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            historico.limpar()
            st.rerun()

    return texto_transcrito


# 8. COMPOSITION ROOT

@st.cache_resource(show_spinner="Carregando modelos...")
def criar_servico(settings: Settings) -> ChatService:
    """Monta o grafo de dependências uma única vez (equivale a um singleton)."""
    return ChatService(
        repositorio=JsonCursoRepository(settings.arquivo_cursos),
        llm=OllamaClient(settings),
        stt=WhisperTranscriber(settings),
        nome_modelo_llm=settings.modelo_llm,
        nome_modelo_stt=settings.descricao_whisper,
    )


def main() -> None:
    """Ponto de entrada do aplicativo Streamlit."""
    # Deve ser a primeira chamada do Streamlit
    st.set_page_config(
        page_title=SETTINGS.titulo_pagina,
        page_icon=SETTINGS.icone_pagina,
        layout="centered",
    )

    servico = criar_servico(SETTINGS)
    historico = Historico()

    texto_transcrito = renderizar_sidebar(servico, historico)

    st.title(f"{SETTINGS.icone_pagina} {SETTINGS.titulo_pagina}")
    st.caption(TEXTO_SUBTITULO)
    renderizar_historico(historico)

    pergunta = st.chat_input("Digite sua pergunta...") or texto_transcrito
    if pergunta:
        processar_pergunta(pergunta, servico, historico)


if __name__ == "__main__":
    main()