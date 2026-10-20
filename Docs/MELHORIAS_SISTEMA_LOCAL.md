# 🛠️ Lista de Melhorias e Evolução do Sistema (100% Local / Sem Agentes)

Este documento reúne propostas de melhorias focadas em **engenharia de visão computacional, robustez operacional, usabilidade e integrações diretas**, operando de forma **100% autônoma, local e sem consumo de APIs externas**.

---

## 1. 📐 Zonas Virtuais de Alarme e Linhas de Cruzamento (*ROI & Tripwires*)

### O que é:
Permite desenhar áreas poligonais ou linhas virtuais sobre o vídeo de cada câmera diretamente na interface web.

### Casos de Uso:
- **Linha do Portão (*Tripwire*)**: Detecta a direção do cruzamento (de fora para dentro = alerta de aproximação; de dentro para fora = saída de morador).
- **Zonas de Exclusão**: Mascara áreas com árvores que balançam com o vento ou a rua pública com tráfego intenso de carros, evitando falsos positivos e economizando processamento.
- **Zona de Proximidade Imediata**: Dispara alertas apenas se o pedestre pisar na área demarcada rente à grade/portão.

### Implementação Técnica:
- Verificação de intersecção ponto-em-polígono via OpenCV (`cv2.pointPolygonTest`) ou Shapely.
- Coordenadas normalizadas salvas no arquivo `config/cameras.json`.

---

## 2. ⏱️ Detecção de Permanência (*Loitering Detection*)

### O que é:
Calcula o tempo contínuo que uma pessoa ou veículo desconhecido fica estacionado ou rondando dentro de uma área crítica.

### Casos de Uso:
- **Pedestre passando reto na calçada**: Registrado como trânsito normal (sem alertas invasivos).
- **Desconhecido parado no portão > 15 segundos**: Disparo de alarme prioritário por atitude suspeita/espera.
- **Veículo parado com motor ligado em frente à garagem > 30 segundos**: Alerta de bloqueio ou monitoramento.

### Implementação Técnica:
- Utiliza os rastros do ByteTrack já existentes no `TrackedEntity`.
- Contador `time_in_zone = now - entry_timestamp`. Disparo de evento ao ultrapassar o limiar configurado.

---

## 3. 📲 Notificações Diretas no Telegram / Discord / WhatsApp (Sem IA)

### O que é:
Envio instantâneo de alertas com foto para o seu smartphone no momento exato da detecção.

### Recursos:
- **Bot do Telegram Nativo**:
  - Envia a foto da evidência (*crop* com zoom ou frame completo anotado).
  - Texto objetivo: `🚨 Alerta: Pessoa Desconhecida no Portão às 23:41`.
  - Botões interativos abaixo da mensagem: `[Ver ao Vivo]` e `[Ignorar]`.
- **Integração Home Assistant (MQTT / Webhook)**:
  - Acender holofotes automaticamente se alguém for detectado no quintal à noite.
  - Tocar chime de campainha interna ao detectar uma pessoa conhecida.

### Implementação Técnica:
- Chamada direta via `httpx` assíncrono para a API oficial de bots do Telegram (`api.telegram.org/bot<TOKEN>/sendPhoto`). Zero latência e zero custo.

---

## 4. 📹 Gravação de Clipes de Vídeo com Pré e Pós-Buffer (*Video Clips MP4*)

### O que é:
Além de salvar snapshots estáticos (JPEG), gravar clipes de vídeo curtos (10 a 20 segundos) em MP4 mostrando o que aconteceu **antes e depois** da detecção.

### Como Funciona:
- O sistema mantém um buffer circular de frames em memória RAM dos últimos 5 segundos.
- Quando o alerta dispara, o sistema salva os 5 segundos anteriores + os 10 segundos posteriores.
- **Resultado**: Você assiste ao vídeo completo do momento em que a pessoa se aproximou, e não apenas uma foto estática.

### Implementação Técnica:
- Codificação acelerada por hardware via GPU NVIDIA (NVENC) usando `ffmpeg-python` ou `cv2.VideoWriter`.

---

## 5. 🏷️ Cadastro Rápido de Pessoas e Veículos Diretamente pela Interface Web

### O que é:
Elimina a necessidade de acessar o terminal para cadastrar novos rostos ou carros.

### Fluxo na Interface:
1. Um desconhecido ou veículo novo aparece na sidebar ou no histórico.
2. Você clica no botão **"Cadastrar como Conhecido"** no modal da evidência.
3. Um formulário abre: você digita o nome (ex: *"Carteiro"*, *"Primo Lucas"*) e opcionalmente a placa.
4. O sistema salva automaticamente o recorte na pasta correspondente (`data/known_faces/` ou `data/known_vehicles/`) e recarrega os embeddings na GPU sem reiniciar o servidor.

---

## 6. 🔊 Alertas Sonoros Configuráveis no Navegador (*Web Audio API*)

### O que é:
Emissão de avisos sonoros diretamente pelos alto-falantes do computador onde o painel estiver aberto.

### Modos de Alerta:
- **Pessoa Conhecida**: *Chime* suave e agradável (notificação de chegada de morador).
- **Veículo Conhecido / Meu Carro**: Tom de confirmação.
- **Desconhecido / Fora de Horário**: Alerta sonoro pulsante de atenção.
- Controle de volume e botão de *Mute* no topo do monitoramento.

---

## 7. 🛡️ Modos de Segurança e Agendamento por Horários (*Arm / Disarm / Night Mode*)

### O que é:
Regras de comportamento do sistema baseadas no horário ou no estado da casa.

### Perfis:
- **Modo Diurno (07:00 às 22:00)**: Alertas discretos; ignora carros passando na rua.
- **Modo Noturno Armado (22:00 às 06:00)**: Sensibilidade máxima; qualquer presença humana nos fundos ou portão gera alerta prioritário com destaque visual em vermelho e envio para o celular.
- **Modo Viagem / Casa Vazia**: Dispara alertas mesmo para eventos de curta duração.

---

## 8. ⚡ Aceleração Máxima com TensorRT na GPU NVIDIA (GTX 1660) `[IMPLEMENTADO ✅]`

### O que é:
Exportar o modelo YOLOv8 do PyTorch para o formato **TensorRT FP16** nativo da NVIDIA.

### Benefícios:
- **Aumento de FPS**: Reduz a latência de inferência de ~15ms para **~3 a 4ms**.
- **Menor consumo de VRAM e energia**: Otimização profunda dos núcleos CUDA da arquitetura Turing da GTX 1660.
- Permite processar mais câmeras simultâneas a 30 FPS estáveis sem elevar a temperatura da placa.

### Como usar no servidor Nobara Linux:
1. Instalar o pacote no ambiente virtual: `pip install tensorrt`
2. Executar o script de compilação e benchmark: `python scripts/export_tensorrt.py`
3. Reiniciar o serviço: `sudo systemctl restart cerberus`

---

## 📦 Exportação de Relatórios de Auditoria e Backup (CSV / PDF / ZIP)

### O que é:
No Dashboard Analítico, adicionar botões de exportação:
- **Exportar CSV/Excel**: Planilha completa de auditoria com horários, câmeras, identidades e placas lidas.
- **Exportar Pacote ZIP de Evidências**: Baixa as fotos selecionadas organizadas por data e pasta para envio a autoridades ou condomínio.

---

## 🏆 Matriz de Priorização Recomendada

| Prioridade | Funcionalidade | Esforço | Impacto Prático no Dia a Dia |
| :---: | :--- | :---: | :--- |
| **Alta** | **Notificações no Telegram com Foto** | Baixo | Receber fotos instantâneas no celular de qualquer lugar |
| **Alta** | **Cadastro Rápido pela Interface Web** | Baixo/Médio | Adicionar novas pessoas e carros com 1 clique no painel |
| **Alta** | **Zonas de Alarme / ROI do Portão** | Médio | Eliminar 100% dos falsos alertas de pedestres na calçada |
| **Média** | **Alertas Sonoros no Painel (Web Audio)** | Baixo | Saber quem chegou pelo som sem precisar olhar para a tela |
| **Média** | **Gravação de Clipes MP4 (Pré/Pós Buffer)** | Médio | Ter o vídeo completo da abordagem e não só fotos |
| **Média** | **Modos de Segurança (Dia/Noite/Armado)** | Baixo | Não incomodar de dia e ser rigoroso na madrugada |
| **Baixa** | **Otimização TensorRT na GTX 1660** | Médio | Desempenho extremo (o YOLOv8 atual já roda muito bem) |
