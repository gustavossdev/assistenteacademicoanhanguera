# 🎓 Assistente Acadêmico

Chatbot que responde dúvidas sobre cursos e matérias de uma faculdade, por **texto** ou **voz**. Roda localmente com [Streamlit](https://streamlit.io), [Ollama](https://ollama.com) (respostas) e [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (transcrição de voz).

As respostas são baseadas apenas na base de conhecimento em `dados/cursos.json`.

## Pré-requisitos

- Python 3.9+
- [Ollama](https://ollama.com/download) instalado
- [FFmpeg](https://ffmpeg.org/download.html) (no Windows: `winget install Gyan.FFmpeg`)

## Instalação e execução

```bash
git clone https://github.com/gustavossdev/assistenteacademicoanhanguera.git
cd assistenteacademicoanhanguera

pip install -r requirements.txt
ollama pull llama3.2

streamlit run app.py
```

Com o Ollama em execução, abra `http://localhost:8501`. Na primeira execução o modelo Whisper é baixado automaticamente.

## Estrutura

```
.
├── app.py            # aplicação completa
├── dados/cursos.json # base de conhecimento
└── requirements.txt
```

## Configuração

Modelos e parâmetros ficam na classe `Settings`, no topo do `app.py` (modelo do Ollama, tamanho do Whisper, idioma, timeout). Para adicionar cursos, edite `dados/cursos.json` e reinicie o app.

## Limitações

- Não há memória de conversa: só a pergunta atual é enviada ao modelo.
- A regra "não invente respostas" é uma instrução ao modelo, não uma garantia.