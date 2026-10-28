# Lista de Perguntas para Teste e Apresentação 🎓

Esta lista contém as perguntas sugeridas para validar e demonstrar o funcionamento do **Assistente Acadêmico**.

As perguntas estão divididas em categorias para garantir a cobertura completa do sistema (recuperação de informação correta e prevenção de alucinações).

---

## 🟢 1. Listagem Geral
Validam se o assistente consegue interpretar a base completa e sumarizar o que existe de macro:
1. *"Quais cursos a faculdade oferece?"*
2. *"Quantos cursos existem?"*
3. *"Quais são as opções de graduação disponíveis?"*

## 🟡 2. Sobre Cursos Específicos
Testam a recuperação de detalhes específicos (duração, descrição) atrelados ao nível do curso:
4. *"O que tem no curso de Ciência da Computação?"*
5. *"Qual a duração do curso de Engenharia de Software?"*
6. *"Me fala sobre o curso de Engenharia de Software."*
7. *"Quantos anos dura o curso de Ciência da Computação?"*

## 🔵 3. Sobre Matérias Específicas
Focam na busca e formatação do conteúdo mais profundo da árvore JSON (matérias e tópicos):
8. *"O que é Processamento Gráfico?"*
9. *"Quais tópicos tem na matéria de Inteligência Artificial?"*
10. *"Me explica o que é Engenharia de Requisitos."*
11. *"O que é OpenGL e em qual matéria eu vejo isso?"* (Verifica se ele associa o tópico à matéria correta)

## 🔴 4. Testes Anti-Alucinação (Casos Negativos)
**MUITO IMPORTANTE:** Estas perguntas devem retornar respostas informando que **o assistente não possui a informação**. Servem para provar que o LLM não está inventando (alucinando) respostas fora da base de conhecimento `cursos.json`.
12. *"Inteligência Artificial tem no curso de Engenharia de Software?"* (Deverá corrigir e dizer que é em Ciência da Computação)
13. *"Existe o curso de Medicina na faculdade?"*
14. *"Qual o valor da mensalidade do curso de Ciência da Computação?"*
15. *"Quem é o professor da matéria de Inteligência Artificial?"*
16. *"Vocês oferecem pós-graduação?"*

---
**Nota para Apresentação:**
Recomendamos gravar um áudio de uma das perguntas da seção 3 ou 4 para demonstrar o pipeline completo (`Faster-Whisper` + `Ollama` com proteção de escopo).
