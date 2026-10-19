# 🐱 🐶 Cadastro de Animais Conhecidos (Pets)

Para cadastrar seus animais de estimação (gatos e cachorros):

1. Crie uma subpasta com o **nome do animal** aqui dentro.
2. Adicione de **3 a 5 fotos** do seu pet (de corpo inteiro, com boa iluminação, mostrando pelagem e manchas).

### Exemplo de Estrutura:
```text
data/known_pets/
├── Meu_Gato_Felix/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── foto3.jpg
└── Meu_Cachorro_Thor/
    ├── foto1.jpg
    └── foto2.jpg
```

- O sistema extrai as características visuais (cores, proporções e pelagem).
- Quando avistado pela câmera, a caixa delimitadora será desenhada em **Roxo/Magenta** com o nome do pet (ex: **`Meu Gato Felix (92%)`**).
- Animais desconhecidos aparecerão como **`Gato Desconhecido #ID`** ou **`Cachorro Desconhecido #ID`**.
