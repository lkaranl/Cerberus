# 📸 Cadastro de Rostos Conhecidos (Face Recognition)

Para cadastrar pessoas para reconhecimento automático nas câmeras:

1. Crie uma subpasta com o **nome da pessoa** aqui dentro.
2. Adicione de **3 a 5 fotos nítidas** do rosto dessa pessoa dentro da pasta.

### Exemplo de Estrutura:
```text
data/known_faces/
├── Karan/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── foto3.jpg
└── Visitante/
    └── foto1.jpg
```

- Formatos aceitos: `.jpg`, `.jpeg`, `.png`
- O nome da pasta será exatamente o nome exibido na tela da câmera e nos alertas (ex: **`Karan (95%)`**).
