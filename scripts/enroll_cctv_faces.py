#!/usr/bin/env python3
"""
Cerberus AI — Treinador Biométrico de Rostos com Fotos Reais de CFTV (Auto-Enrollment)
Objetivo: Permite listar os eventos com rostos capturados pelas câmeras que não foram
identificados e cadastrá-los instantaneamente no perfil dos moradores (Karan, Ingrid, etc.),
ensinando a rede neural ArcFace com os ângulos, iluminação e ótica reais do local.
"""

import sys
import os
import shutil
import sqlite3
from pathlib import Path

# Ajusta path para importar módulos do Cerberus
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    print("=" * 65)
    print(" 👤 CERBERUS AI — TREINAMENTO BIOMÉTRICO COM FOTOS DE CFTV")
    print("=" * 65)

    db_path = Path("data/events.db")
    if not db_path.exists():
        print(f"[!] Banco de dados não encontrado em {db_path}.")
        print("[!] Certifique-se de executar no servidor onde o Cerberus está operando.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Busca pessoas cadastradas atualmente
    known_dir = Path("data/known_faces")
    known_persons = []
    if known_dir.exists():
        known_persons = sorted([d.name for d in known_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    print(f"[*] Moradores registrados na galeria: {', '.join(known_persons) if known_persons else 'Nenhum'}")
    print()

    # Busca os últimos 20 eventos de pessoas não identificadas
    cursor.execute("""
        SELECT event_id, camera_name, timestamp, crop_path, full_path, confidence
        FROM events
        WHERE entity_type = 'person' AND is_known = 0
        ORDER BY timestamp DESC
        LIMIT 25
    """)
    rows = cursor.fetchall()

    if not rows:
        print("[✓] Nenhuma detecção desconhecida encontrada no banco de dados!")
        return

    print(f"[+] Encontrados {len(rows)} rostos recentes não identificados:")
    valid_events = []
    for idx, row in enumerate(rows, start=1):
        crop = Path(row["crop_path"])
        exists = "✓" if crop.exists() else "✗ (arquivo ausente)"
        print(f"  [{idx:02d}] Evento {row['event_id'][:8]} | {row['camera_name']} | {row['timestamp']} | Arquivo: {exists}")
        if crop.exists():
            valid_events.append((idx, row))

    if not valid_events:
        print("\n[!] Nenhum arquivo de imagem das detecções foi encontrado no disco.")
        return

    print("\n" + "-" * 65)
    print(" Como deseja classificar?")
    print(" 1. Classificar uma detecção específica pelo número")
    print(" 2. Atribuir as últimas N detecções em lote para um morador")
    print(" 0. Sair")
    print("-" * 65)

    choice = input("Escolha uma opção: ").strip()

    if choice == "1":
        num_str = input("Digite o número da detecção (ex: 1): ").strip()
        try:
            sel_num = int(num_str)
            target = next((r for n, r in valid_events if n == sel_num), None)
            if not target:
                print("[!] Número inválido.")
                return
            
            print(f"\nMoradores conhecidos: {', '.join(known_persons)}")
            person_name = input("Digite o nome da pessoa (ex: Karan): ").strip()
            if not person_name:
                print("[!] Nome inválido.")
                return

            _enroll_single_event(conn, target, person_name)
        except ValueError:
            print("[!] Entrada inválida.")
            return

    elif choice == "2":
        person_name = input(f"Digite o nome do morador ({', '.join(known_persons)}): ").strip()
        if not person_name:
            print("[!] Nome inválido.")
            return

        count_str = input(f"Quantos dos últimos eventos atribuir a '{person_name}'? (1 a {len(valid_events)}): ").strip()
        try:
            limit = int(count_str)
            for _, target in valid_events[:limit]:
                _enroll_single_event(conn, target, person_name)
            print(f"\n[✓] {min(limit, len(valid_events))} fotos associadas com sucesso a '{person_name}'!")
        except ValueError:
            print("[!] Número inválido.")
            return

    # Notifica sobre a atualização da biometria
    print("\n[⚡] Atualizando galeria biométrica do Cerberus...")
    try:
        from app.detector.recognizer import face_recognizer
        face_recognizer.load_known_faces()
        print(f"[✓] Galeria recarregada! Total de vetores ativos: {face_recognizer.gallery_vector_count}")
    except Exception as e:
        print(f"[*] Para aplicar imediatamente no serviço em execução, rode:")
        print(f"    sudo systemctl restart cerberus.service")

    conn.close()
    print("\n" + "=" * 65)
    print(" Concluído! O Cerberus agora reconhecerá essa pessoa com maior precisão.")
    print("=" * 65)

def _enroll_single_event(conn, event_row, person_name: str):
    clean_name = person_name.strip()
    dest_dir = Path("data/known_faces") / clean_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    src_crop = Path(event_row["crop_path"])
    event_id = event_row["event_id"]
    dest_file = dest_dir / f"cctv_{event_id[:8]}.jpg"

    shutil.copyfile(src_crop, dest_file)
    print(f"  [+] Foto copiada para: {dest_file}")

    cursor = conn.cursor()
    cursor.execute("""
        UPDATE events
        SET person_name = ?, is_known = 1
        WHERE event_id = ?
    """, (clean_name, event_id))
    conn.commit()

if __name__ == "__main__":
    main()
