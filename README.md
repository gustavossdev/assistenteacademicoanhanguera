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
```

### Ollama (obrigatório)

O assistente só responde se o Ollama estiver **em execução** e com o **modelo `llama3.2` baixado**. Rode no terminal:

```bash
ollama pull llama3.2
ollama run llama3.2
```

O `pull` baixa o modelo (só na primeira vez) e o `run` o inicia. Quando aparecer o prompt de conversa, digite `/bye` para sair do chat: o Ollama continua rodando em segundo plano.

Para conferir se o modelo foi baixado, use `ollama list`.

### Iniciar o assistente

```bash
streamlit run app.py
```

Abra `http://localhost:8501`. Na primeira execução o modelo Whisper também é baixado automaticamente, então a abertura pode demorar um pouco.

Se aparecer a mensagem "Não consegui conectar ao Ollama", ele não está em execução.

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
