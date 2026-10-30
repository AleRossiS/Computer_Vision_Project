import cv2
from ultralytics import YOLO

def main():
    # 1. Carica il modello pre-addestrato per la Pose Estimation.
    # Usiamo la versione "Medium" (yolov8m-pose.pt) per maggiore precisione
    # al primo avvio ci metterà qualche secondo a scaricare i pesi del modello.
    model = YOLO("yolov8m-pose.pt")

    # 2. Apri il video rettificato di test
    video_path = "../videos/hpe_01/out1.mp4" # Assicurati che questo percorso sia corretto!
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire il video: {video_path}")
        return

    # Preparazione per salvare il video di output
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = "yolo_test_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Inizio elaborazione con il modello {model.ckpt_path.split('/')[-1]}...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 3. Esegui YOLO sul singolo frame
        # conf=0.25: Disegna lo scheletro anche se è sicuro solo al 25% che sia una persona
        results = model(frame, conf=0.50, imgsz=1280, verbose=False)
        
        # --- DEBUG: Controlliamo quante persone ha trovato in questo frame ---
        num_persone = len(results[0].boxes) if results[0].boxes is not None else 0
        
        # 4. Disegna automaticamente lo scheletro sul frame
        # kpt_radius=4 fa i "pallini" delle articolazioni più grandi
        annotated_frame = results[0].plot(kpt_radius=4, line_width=2)
        
        # Salva il frame nel nuovo video
        out.write(annotated_frame)

        # (Opzionale) Estrazione dei dati numerici che ti serviranno per la triangolazione 3D
        # keypoints.xy contiene le coordinate [x, y] di tutte le articolazioni
        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            # In futuro scriveremo qua il codice per salvare i keypoints su file
            pass

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   -> Frame {frame_count}: Trovate {num_persone} persone in campo")

    cap.release()
    out.release()
    print(f"\n[OK] Finito! Guarda il video con gli scheletri in: {output_path}")

if __name__ == "__main__":
    main()