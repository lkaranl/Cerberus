# 🤖 Propostas de Melhorias com Agente de IA (OpenRouter)

Este documento reúne sugestões práticas e de alto valor agregado para integrar um **Agente de IA** ao **Cerberus (Sentinela Vision)** utilizando a **API da OpenRouter**.

> **Diretriz de Arquitetura**:
> - **Baixo Consumo de Tokens**: Prompts enxutos, modelos rápidos e econômicos (ex: `google/gemini-2.5-flash`, `deepseek/deepseek-chat`, `meta-llama/llama-3.1-8b-instruct` ou `qwen/qwen-2.5-7b-instruct`).
> - **Disparo por Evento (Event-Driven)**: A IA **nunca** processa frames contínuos a 30 FPS. As chamadas acontecem apenas em momentos-chave (ao registrar um alerta importante, no fechamento do dia ou quando o usuário fizer uma pergunta).
> - **Privacidade e Eficiência**: O processamento pesado de visão computacional (YOLOv8, FaceNet e LPR) continua 100% local e gratuito na GPU. Apenas metadados em texto (ou um crop pontual) vão para a nuvem.

---

## 1. 🛡️ Avaliador de Suspeição e Risco Contextual (Heurística Inteligente)

### O Problema:
Um alerta de "Pessoa Desconhecida" ao meio-dia na calçada é comum, mas o mesmo alerta às 03:15 da madrugada na câmera dos fundos é crítico.

### Como o Agente Resolve:
Quando um evento ocorre em condições de potencial risco, o backend envia um JSON compacto (~100 tokens):
```json
{
  "camera": "Área do Fundo",
  "horario": "03:22",
  "tipo": "pessoa",
  "identidade": "Desconhecido",
  "tempo_na_cena_seg": 38,
  "objetos_proximos": ["portao"]
}
```

- **Prompt do Sistema**: *"Você é um auditor de segurança residencial. Analise os metadados do evento e responda em JSON com nível de risco (baixo, medio, alto) e uma frase de ação."*
- **Resposta do Agente** (~40 tokens):
  ```json
  {"risco": "alto", "motivo": "Permanência de desconhecido em área vulnerável durante a madrugada.", "acao": "Verificar câmeras imediatamente."}
  ```
- **Consumo**: ~150 tokens por alerta crítico.

---

## 2. 💬 Chat / Assistente em Linguagem Natural sobre o Histórico

### O Problema:
Para saber se alguém esteve em casa, o usuário precisa rolar tabelas de histórico ou filtrar logs manualmente.

### Como o Agente Resolve:
Adicionar um campo de busca/chat na interface web:
- **Pergunta do Usuário**: *"O carteiro ou alguém de moto passou hoje de manhã?"*
- **Fluxo do Sistema**:
  1. O backend filtra no SQLite os eventos de hoje com tipo `vehicle` (moto) ou `person`.
  2. Monta uma lista resumida em poucas linhas (~120 tokens).
  3. O Agente sintetiza a resposta humana em 1 frase:
     > *"Sim, uma moto foi detectada no Portão às 10:42 da manhã com permanência de 2 minutos."*
- **Consumo**: ~200 tokens por pergunta feita pelo usuário.

---

## 3. 📋 Briefing Executivo Diário / Resumo de Segurança

### O Problema:
Centenas de eventos ao longo do dia geram fadiga de notificação.

### Como o Agente Resolve:
Uma vez ao dia (ex: 22h ou 07h da manhã), o sistema coleta os números agregados do banco de dados SQLite e solicita um resumo amigável:
- **Entrada do Agente**:
  ```text
  Data: 19/10/2026. Total eventos: 42.
  Karan detectado: 4x (chegada 18:15).
  Meu Carro: estacionado desde 18:20.
  Desconhecidos: 3 pedestres no portão (horários: 09:10, 14:30, 17:05).
  Pets: Rex ativo às 08h e 16h.
  Nenhum evento entre 00h e 06h.
  ```
- **Saída do Agente** (~60 tokens):
  > *"Resumo do dia: Dia calmo com atividade externa concentrada no horário comercial. Seu carro estacionou às 18:20. Nenhuma movimentação anormal na madrugada."*
- **Consumo**: ~150 tokens por dia (menos de R$ 0,01 por mês).

---

## 4. 📲 Enriquecimento de Notificações (Telegram / WhatsApp / Discord)

### O Problema:
Mensagens automáticas como `[ALERTA] cam-0 person id#12` são frias e não transmitem o contexto de relance.

### Como o Agente Resolve:
O agente traduz os parâmetros técnicos para uma mensagem em linguagem natural e direta:
- **Técnico**: `camera=Garagem, entity=vehicle, plate=NEC4J87, status=estacionado`
- **Mensagem da IA**: *"🚗 Seu carro acabou de estacionar na Garagem."*
- **Técnico**: `camera=Portao, entity=person, name=Desconhecido, time=11:30, duration=15s`
- **Mensagem da IA**: *"👤 Uma pessoa desconhecida passou em frente ao Portão."*
- **Consumo**: ~60 tokens por mensagem formatada.

---

## 5. 🏷️ Categorização Semântica de Atividades

### O Problema:
O detector sabe apenas as classes básicas ("Carro", "Pessoa", "Moto").

### Como o Agente Resolve:
Combinando 2 ou 3 detecções sequenciais com um prompt mínimo de classificação (`temperature: 0.1`):
- `Moto` + `Pessoa com capacete/objeto` ➔ **"Provável Entrega / Delivery"**
- `Carro conhecido` + `Pessoa saindo` ➔ **"Chegada em Casa"**
- `Pessoa desconhecida` parada > 2 min ➔ **"Espera / Monitoramento"**
- Essa etiqueta é salva no banco SQLite para buscas posteriores com zero custo computacional na GPU local.

---

## 6. 👁️ VLM sob Demanda para Tirar-Dúvidas (Opcional / Alta Criticidade)

### O Problema:
Um objeto não identificado ou uma sombra gerou dúvida no detector YOLO durante a noite.

### Como o Agente Resolve:
Apenas quando o usuário clicar no botão *"Pedir Análise da IA"* no modal do dashboard:
- O frontend redimensiona o recorte para 256x256 (JPEG leve de 15KB).
- Envia via OpenRouter para um Vision Model econômico (`google/gemini-2.5-flash` ou `qwen/qwen-2.5-vl-7b-instruct`).
- **Prompt**: *"Descreva o que há na imagem em 10 palavras focando em segurança."*
- **Resposta**: *"Pessoa de casaco escuro segurando uma mochila próximo à grade."*
- **Consumo**: Chamada 100% sob demanda manual do usuário.

---

## 📊 Tabela de Estimativa de Consumo e Custos na OpenRouter

| Funcionalidade | Frequência Típica | Tokens Médios | Custo Mensal Estimado (Gemini Flash / Llama 8B) |
| :--- | :--- | :--- | :--- |
| **Resumo Diário Executivo** | 1x ao dia | ~200 tokens | < **R$ 0,05 / mês** |
| **Análise de Risco na Madrugada** | 2 a 5x por noite | ~150 tokens | < **R$ 0,10 / mês** |
| **Chat de Histórico (Dúvidas)** | 3 a 5 consultas/dia | ~250 tokens | < **R$ 0,15 / mês** |
| **Notificações Humanizadas** | Eventos selecionados | ~80 tokens | < **R$ 0,20 / mês** |
| **Total Estimado** | Operação 24/7 | Baixíssimo | **Inferior a R$ 1,00 / mês** |

---

## 🛠️ Próximos Passos Sugeridos para Implementação
1. Adicionar `OPENROUTER_API_KEY` e `OPENROUTER_MODEL` no arquivo `.env`.
2. Criar um módulo cliente leve assíncrono em `app/agent/client.py` usando `httpx` (sem SDKs pesados).
3. Integrar o Resumo Diário no Dashboard e o Auditor de Risco Noturno.
