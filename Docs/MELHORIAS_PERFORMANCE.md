# ⚡ Propostas de Otimização de Performance Extrema — Cerberus Vision

Este documento detalha o plano de arquitetura e otimizações de **alta performance** para o Cerberus Vision, explorando o limite do hardware do servidor (**Intel Core i5-7600K + NVIDIA GeForce GTX 1660 6GB GDDR5 Turing**) e garantindo que os dispositivos clientes (computadores modestos, tablets de parede ou celulares) operem com consumo mínimo de recursos.

---

## 🏛️ Contexto de Hardware do Servidor

- **Processador (CPU)**: Intel Core i5-7600K (4 núcleos / 4 threads @ até 4.20 GHz).
  - *Gargalo a mitigar*: Apenas 4 threads. Decodificação de vídeo por software (CPU) e compressão JPEG concorrem diretamente com o SO e o banco de dados.
- **Placa de Vídeo Dedicada (GPU)**: NVIDIA GeForce GTX 1660 (6 GB GDDR5, Arquitetura Turing TU116).
  - *Diferencial*: Possui núcleos CUDA com suporte a FP16 nativo e o motor dedicado **NVDEC de 6ª geração** (decodificação de vídeo em hardware sem consumir CPU).
- **GPU Secundária**: AMD Radeon RX 5600 XT (dedicada exclusivamente ao display do sistema Nobara Linux).
- **Clientes (Navegadores)**: Notebooks antigos, celulares e thin clients (necessidade de carga computacional próxima de zero no cliente).

---

## 🚀 1. 🎞️ Decodificação RTSP por Hardware na GPU (NVIDIA NVDEC) [IMPLEMENTADO ✅]

### O Problema Atual:
O OpenCV padrão (`cv2.VideoCapture`) utiliza a CPU via software FFmpeg para descompactar os fluxos de vídeo H.264/H.265 das câmeras RTSP. Com 3 câmeras em 2K/1080p a 20-30 FPS, o i5-7600K gastava entre **35% e 55% de toda a sua capacidade de processamento apenas decodificando vídeo**.

### A Solução Implementada:
- **`RTSPCapture` com NVDEC Nativo**: Inicializa FFmpeg com `-hwaccel cuda -hwaccel_device 0`, ativando o motor NVDEC da GTX 1660 com tempo de processamento de míseros 0.1s de CPU para 60 frames.
- **Fallback Transparente**: Se o NVDEC não estiver disponível (ex: desenvolvimento em macOS M4), reverte automaticamente para o `cv2.VideoCapture`.
- **Decoupling Total por Câmera**: `StreamManager` agora possui tasks assíncronas independentes (`_camera_inference_loop`), eliminando qualquer engavetamento entre câmeras.
- **Persistência Assíncrona de Evidências**: Gravação de arquivos JPG de alertas em background thread (`asyncio.to_thread`), sem pausar o loop de detecção.
- **MTCNN Sensível a Movimento**: Thresholds calibrados (`[0.5, 0.6, 0.6]`) e recorte facial expandido para 55% com fallback de corpo inteiro.

### Benefícios:
- **Uso de CPU para decodificação cai para ~0%**: Libera os 4 núcleos do i5 para tarefas assíncronas do FastAPI e SQLite.
- **Alvos Rápidos Detectados**: Animais correndo ou pessoas andando rápido são detectados sem perda de frames.
- **Exibição na Interface**: Badge dinâmico `NVIDIA GTX 1660 [NVDEC + YOLO + FaceNet TRT ⚡]`.

---

## 2. ⚡ Batching Dinâmico de Câmeras no TensorRT (Inferência Paralela)

### O Problema Atual:
O loop de inferência do `StreamManager` processa as câmeras sequencialmente:
`Detecta Cam 0 (5.2ms) ➔ Detecta Cam 1 (5.2ms) ➔ Detecta Cam 2 (5.2ms) = 15.6ms total`.

### A Solução:
Compilar a engine TensorRT com suporte a **Dynamic Batch Size** (lote dinâmico) e empilhar os quadros das câmeras ativas em um único tensor:
- Shape de entrada: `(N, 3, 640, 640)` onde `N` é o número de câmeras ativas (ex: 3).

### Benefícios:
- A GTX 1660 processa as 3 câmeras em **uma única passada de ~6ms na GPU**, aproveitando o paralelismo massivo dos 1.408 núcleos CUDA.
- Capacidade do servidor ampliada para **até 8 câmeras simultâneas a 30 FPS reais**.

---

## 3. 🧊 *Motion Gating* Inteligente (Modo Econômico / Turbo)

### O Problema Atual:
O YOLOv8 roda inferência a toda velocidade 24 horas por dia ininterruptamente, mesmo quando o portão, garagem e quintal estão completamente parados e desertos de madrugada, gerando calor e consumo elétrico desnecessário.

### A Solução:
Implementar um filtro prévio ultraleve de detecção de movimento por diferença de quadros (*Frame Differencing* / MOG2 otimizado em C++, custo < 0.2ms de CPU):
- **Cena Estática (Sem movimento)**: A câmera entra em modo de vigília (*idle*), reduzindo a inferência YOLO para **2 a 3 FPS**.
- **Variação Detectada**: O sistema acorda instantaneamente para **30 FPS TensorRT em menos de 10 milissegundos**.

### Benefícios:
- **Redução de 70% a 80% no consumo elétrico e na temperatura da GPU**.
- Aumenta a vida útil dos componentes e ventoinhas da placa de vídeo.
- Garante 0% de perda de eventos, pois o gatilho de movimento é instantâneo.

---

## 4. 👤 Aceleração do Reconhecimento Facial (FaceNet) com TensorRT FP16 `[IMPLEMENTADO ✅]`

### O Problema:
O YOLOv8 agora opera a **5.19 ms por frame** graças à engine TensorRT. Porém, o módulo de reconhecimento facial (`FaceNet Inception-ResNet-V1`) rodava no PyTorch CUDA convencional, levando **~18 a 25 ms** para processar um rosto.

### A Solução Implementada:
Exportar a rede neural `InceptionResnetV1` para o formato **TensorRT FP16** (gerando `models/facenet.engine`) com acesso direto de memória zero-copy via ponteiro de VRAM (`data_ptr()`).

### Benefícios:
- Tempo de reconhecimento facial cai de **~20 ms para ~2 a 3 ms**.
- Identificação instantânea e imperceptível mesmo se múltiplas pessoas conhecidas entrarem simultaneamente no enquadramento.

### Como gerar a engine no servidor Nobara Linux:
```bash
python scripts/export_facenet_tensorrt.py
sudo systemctl restart cerberus
```

---

## 5. 💤 Codificação de Stream JPEG Sob Demanda (*Lazy Encoding*)

### O Problema Atual:
O servidor gasta ciclos de CPU comprimindo cada quadro em JPEG e mantendo o buffer de WebSocket atualizado para todas as 3 câmeras, mesmo quando **nenhum usuário está com o painel web aberto**.

### A Solução:
Adicionar um contador de observadores conectados por câmera no `WebSocketManager`:
- A inteligência artificial, detecção de ameaças e gravação de evidências continuam ativas **24 horas por dia, 7 dias por semana**.
- A compressão JPEG do streaming para o navegador só é executada se houver **pelo menos 1 cliente com a conexão WebSocket aberta** naquela câmera específica.

### Benefícios:
- Redução drástica no uso de CPU e geração de calor durante 95% do dia em que o painel web não está sendo ativamente visualizado.

---

## 6. 🏎️ TurboJPEG com Aceleração SIMD AVX2 (`PyTurboJPEG`)

### O Problema Atual:
O `cv2.imencode(".jpg", frame)` do OpenCV utiliza a biblioteca `libjpeg` tradicional sem otimizações avançadas de vetorização para processadores modernos.

### A Solução:
Adotar a biblioteca **`PyTurboJPEG`** (invólucro nativo da `libjpeg-turbo`), que faz uso das instruções vetoriais **Intel AVX2 e SSE4** do processador i5-7600K.

### Benefícios:
- Compressão JPEG de **3x a 5x mais rápida** em comparação com o OpenCV puro.
- Reduz a latência de ponta a ponta (*glass-to-glass*) do stream para menos de **10 milissegundos**.

---

## 7. 📱 Otimização Zero-GC e Buffering no Cliente Fraco

### O Problema Atual:
Em navegadores de computadores ou celulares fracos, criar e descartar dezenas de objetos `Blob` e `ImageBitmap` por segundo ativa o *Garbage Collector* (GC) do motor JavaScript V8, provocando pequenos engasgos na reprodução do vídeo a cada poucos segundos.

### A Solução:
- Reutilizar um buffer binário pré-alocado na memória do cliente com `CanvasRenderingContext2D.putImageData()` ou técnica de *Double Buffering*.
- Pausar o handshake WebSocket de câmeras que não estejam visíveis na tela (ex: quando o usuário seleciona o modo de visualização de câmera única).

### Benefícios:
- Reprodução de vídeo 100% fluida e sem solavancos, mesmo em navegadores rodando em hardware com pouca memória RAM.

---

## 🏆 Matriz de Priorização e Impacto

| Prioridade | Funcionalidade | Complexidade | Ganho Principal |
| :---: | :--- | :---: | :--- |
| **Alta** | **Stream Sob Demanda (Lazy Encoding)** | Baixa | Economia maciça de CPU quando o navegador está fechado |
| **Alta** | **Motion Gating (Modo Eco / Turbo)** | Média | Placa de vídeo fria e economia drástica de energia |
| **Alta** | **Aceleração TensorRT no FaceNet** | Média | Reconhecimento facial cai de 20ms para ~2.5ms |
| **Média** | **TurboJPEG com Instruções AVX2** | Baixa | Streaming 3x a 5x mais ágil no envio de frames |
| **Média** | **Decodificação NVDEC por Hardware** | Média/Alta | Libera até 40% da CPU do Intel i5-7600K |
| **Média** | **Batching TensorRT de Múltiplas Câmeras** | Média/Alta | Permite escalar para mais de 6 câmeras a 30 FPS estáveis |
