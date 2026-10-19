# 🚗 Cadastro de Veículos Conhecidos

Para cadastrar seus veículos (carros, caminhões, motos):

1. Crie uma subpasta com o **nome do veículo** e, opcionalmente, a **placa entre colchetes** no nome da pasta.
2. Adicione de **3 a 5 fotos** do veículo (frente com placa visível, traseira e lateral).

### Exemplo de Estrutura:
```text
data/known_vehicles/
├── Meu_Carro_[BRA2E19]/
│   ├── frente.jpg
│   ├── lateral.jpg
│   └── traseira.jpg
└── Moto_Honda_[ABC1234]/
    └── foto1.jpg
```

- Quando a câmera detectar o veículo:
  - Se ler a placa (`BRA2E19`), confirmará a identidade instantaneamente.
  - Se a placa não estiver legível, usará a comparação visual das fotos (cor e silhueta).
- As caixas delimitadoras aparecerão em **Azul Ciano** com o nome e a placa.
