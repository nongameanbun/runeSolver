import os
import threading
import pickle
from ultralytics import YOLO
from fastapi import FastAPI
import uvicorn
import torch
import dotenv

from utils.rune_utils import *
from gateway import *


dotenv.load_dotenv()

port_number = int(os.getenv("RUNE_SOLVER_PORT", "8020"))

app = FastAPI(title = "Rune Solver API", description = "API for solving rune")

class YoloModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            # os.chdir("yolov5")
            cls._instance = super(YoloModel, cls).__new__(cls)
            cls._instance.rune_model = YOLO('src/rune_250331.pt')
            cls._instance.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return cls._instance

    
    def detect_v5(self, img):
        ## detect image
        detect_res = self.rune_model(img, conf= 0.8, verbose=False)[0].boxes
        detect_final = torch.cat((detect_res.xywhn, detect_res.data), dim= 1)

        # ## sort by confidence
        res_sorted_conf = sorted(detect_final, key=lambda x: x[8], reverse= True)[:4]

        # ## sort by x position
        res_sorted_x = sorted(res_sorted_conf, key=lambda x: x[4], reverse= False)

        res_converter = ['left', 'down', 'right', 'up', 'rotate']
        final_res = []
        total = []
        centers = []

        rotate_index = []
        for i in range(4):
            try :
                val = int(res_sorted_x[i][9].item())
                final_res.append(res_converter[val])
                total.append((val, res_sorted_x[i][0:4].tolist()))
                if val == 4:
                    rotate_index.append(i)
            except :
                pass
        
        for i in range(len(total)) :
            centers.append((total[i][1][0] * 460, total[i][1][1] * 135))

        return {"res" : final_res, "centers" : centers, "rotate_index" : rotate_index}
    
    def device_check(self):
        return self.device

yolo = YoloModel()



def solver() :
    # record for rotation rune
    rotation_rune_thread = threading.Thread(target=rune_video)
    rotation_rune_thread.start()

    normal_res, rotate_index, centers = get_initial_answer(yolo)

    if normal_res == None :
        return None

    # normal rune
    if (len(rotate_index) == 0):        
        return normal_res
        
    ## rotation rune
    rotation_rune_thread.join()

    send_message("Rotation Detected")
    
    flag = False
    for _ in range(10):
        try:
            with open("rune_video.pkl", "rb") as f:
                video = pickle.load(f)
            flag = True
            break
        except Exception as e:
            print(e)
            continue
    if not flag:
        return None


    last_angle = [-1, -1, -1, -1]
    after_chulkuk_angle = [[], [], [], []]
    for _, img in enumerate(video):
        crop = masking(img)

        for j, (x_center, y_center) in enumerate(centers):
            if j in rotate_index:
                ang = get_angle(crop, x_center, y_center)

                if( last_angle[j] != -1):
                    if(last_angle[j] < ang and 6 < abs(ang - last_angle[j]) < 250):
                        after_chulkuk_angle[j].append(ang)
                    elif(last_angle[j] > ang and abs(ang - last_angle[j]) > 250):
                        after_chulkuk_angle[j].append(ang)
                last_angle[j] = ang

    rotation_answer = chulkuk_parser(after_chulkuk_angle)
    # print(rotation_answer) # ex. [-1, 90, -1, 180]
    
    final_ans = []
    converter = {0 : "right", 90 : "up", 180 : "left", 270 : "down", 360 : "right"}
    for i in range(4) :
        final_ans.append(converter.get((rotation_answer[i]), normal_res[i]))
    return final_ans


@app.get("/solve_rune", description = "Solve the rune and return the answer")  
async def solve_rune():
    try :
        answer = solver()
        return {"resp" : answer}
    except Exception as e:
        return {"resp" : f"Error:{str(e)}"}

@app.post("/awake_model", description = "Awake the model to reduce the first inference time")
async def awake_model():
    threading.Thread(target=yolo.detect_v5, args=("src/awake.jpg",)).start()
    return {"resp": "Model awakened"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port_number, log_level="warning")