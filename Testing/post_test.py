import onnxruntime as rt
import numpy as np
import cv2

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
 
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(img):
    model_path = "./model/"
    model_name = "YOLOv5s.onnx"

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img,(640,640))/255.

    img2 = np.transpose(img2, axes=[2,0,1])
    print(img2.shape)
    img2 = np.reshape(img2, (1,3,640,640))
    sess = rt.InferenceSession(model_path+model_name)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run(\
        [label_name], {input_name: img2.astype(np.float32)})
    nparr = np.array(pred)
    print(nparr.shape)
    return pred

if __name__ == "__main__":
    frame = cv2.imread('bus.jpg')
    detections = pre_process(frame)
    rows = detections[0].shape[1]
    image_height, image_width = frame.shape[:2]
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    print(rows, image_height, image_width, x_factor, y_factor)
    class_ids = []
    confidences = []
    boxes = []
    for r in range(rows):
            row = detections[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= CONFIDENCE_THRESHOLD:
                  classes_scores = row[5:]
                  # Get the index of max class score.
                  class_id = np.argmax(classes_scores)
                  #  Continue if the class score is above threshold.
                  if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)
    
    #NMS for 1 bounding box
    score = confidences.copy()
    score = np.argsort(np.array(score))[::-1]
    indices = list()
    # boxes = np.reshape(boxes)
    print(len(confidences))
    boxes = np.array(boxes).T
    # print()np.reshape(boxes[:2,score[0], (2,1)
    print("score.shape", len(score))
    scores = score
    while len(score) > 0:
        current_index = score[0]
        indices.append(current_index)
        #iou

        upper_left = np.maximum(boxes[:2,score[1:]]- ((boxes[2:,score[1:]])/2), np.reshape(boxes[:2,current_index]- boxes[2:,current_index]/2, (2,1)))
        lower_right = np.minimum(boxes[:2,score[1:]]+ boxes[2:,score[1:]]/2, np.reshape(boxes[:2,current_index] + boxes[2:,current_index]/2,(2,1)))
        w_h_rect = np.prod(boxes[2:,score[:]], axis = 0, keepdims=1)

        intersection_rect = np.maximum(0,lower_right - upper_left)
        area_inter = np.prod(intersection_rect,axis=0, keepdims=1)
        area_union = (w_h_rect[:,1:] + w_h_rect[:,0]) - area_inter
        iou =np.divide( area_inter[0], area_union[0] )
        #end iou
        remaining_indices = score[1:][iou <= NMS_THRESHOLD]
        score = scores[remaining_indices]
      
        print('iou',iou.shape)
        print('remaiing', remaining_indices.shape)
        print('score',score.shape)
    print('boundin', len(indices))
        
        
  