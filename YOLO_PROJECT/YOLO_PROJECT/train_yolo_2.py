from ultralytics import YOLO

def main():

    model = YOLO('yolov8s.pt') 

  
    results = model.train(
        data='/home/linsx/lung_1_YOLO_3/lung.yaml',  
        epochs=100,            
        imgsz=640,              
        batch=16,                
        device='0',            
        project='/home/linsx/lung_1_YOLO_3/runs', 
        name='lung_cancer_exp', 
        patience=50,            
        workers=8,             
        cache=True              
    )

    print("Training Finished!")
    print(f"Results saved to {results.save_dir}")

if __name__ == '__main__':
    main()