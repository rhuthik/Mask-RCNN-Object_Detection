import cv2 as cv
import argparse
import numpy as np
import os
import sys
import random

class MaskRCNN:
    def __init__(self, args):
        self.confThreshold = 0.5
        self.maskThreshold = 0.3
        self.classes = self.load_classes("mscoco_labels.names")
        self.colors = self.load_colors("colors.txt")
        self.network = self.initialize_network(args.device)
        self.capture_source = self.setup_capture_source(args)
        self.output_file = self.setup_output_file(args)

    def load_classes(self, filename):
        with open(filename, 'rt') as f:
            return f.read().rstrip('\n').split('\n')

    def load_colors(self, filename):
        with open(filename, 'rt') as f:
            return [np.array(color.split(' '), dtype=np.float32) for color in f.read().rstrip('\n').split('\n')]

    def initialize_network(self, device):
        text_graph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        model_weights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
        net = cv.dnn.readNetFromTensorflow(model_weights, text_graph)
        if device == "cpu":
            net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        elif device == "gpu":
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        return net

    def setup_capture_source(self, args):
        if args.image:
            if not os.path.isfile(args.image):
                sys.exit(f"Input image file {args.image} doesn't exist")
            return cv.VideoCapture(args.image)
        elif args.video:
            if not os.path.isfile(args.video):
                sys.exit(f"Input video file {args.video} doesn't exist")
            return cv.VideoCapture(args.video)
        else:
            return cv.VideoCapture(0)

    def setup_output_file(self, args):
        if args.image:
            return args.image[:-4] + '_mask_rcnn_out_py.jpg'
        elif args.video:
            return args.video[:-4] + '_mask_rcnn_out_py.avi'
        return "mask_rcnn_out_py.avi"

    def run(self):
        video_writer = None
        if self.output_file.endswith('.avi'):
            # Initialize video writer for video files
            fourcc = cv.VideoWriter_fourcc(*'MJPG')
            fps = self.capture_source.get(cv.CAP_PROP_FPS)  # Use source's FPS
            frame_size = (int(self.capture_source.get(cv.CAP_PROP_FRAME_WIDTH)),
                          int(self.capture_source.get(cv.CAP_PROP_FRAME_HEIGHT)))
            video_writer = cv.VideoWriter(self.output_file, fourcc, fps, frame_size)

        while True:
            has_frame, frame = self.capture_source.read()
            if not has_frame:
                break

            blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)
            self.network.setInput(blob)
            boxes, masks = self.network.forward(['detection_out_final', 'detection_masks'])
            self.postprocess(frame, boxes, masks)

            if video_writer is not None and has_frame:
                video_writer.write(frame)  # Write frame to video

            cv.imshow('Mask-RCNN', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        if video_writer is not None:
            video_writer.release()
        self.capture_source.release()
        cv.destroyAllWindows()

    def postprocess(self, frame, boxes, masks):
        height, width = frame.shape[:2]
        for i in range(boxes.shape[2]):
            box = boxes[0, 0, i]
            mask = masks[i]
            score = box[2]
            if score > self.confThreshold:
                classId = int(box[1])
                left, top, right, bottom = self.getBoundingBox(height, width, box)
                classMask = mask[classId]
                self.drawPred(frame, classId, score, left, top, right, bottom, classMask)

    def getBoundingBox(self, height, width, box):
        left = int(width * box[3])
        top = int(height * box[4])
        right = int(width * box[5])
        bottom = int(height * box[6])
        left, top, right, bottom = [max(0, min(coord, dim - 1)) for coord, dim in zip([left, top, right, bottom], [width, height, width, height])]
        return left, top, right, bottom

    def drawPred(self, frame, classId, conf, left, top, right, bottom, classMask):
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        
        label = '%.2f' % conf
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)
        
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

        # Mask processing
        classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
        mask = (classMask > self.maskThreshold)
        roi = frame[top:bottom + 1, left:right + 1][mask]

        colorIndex = random.randint(0, len(self.colors) - 1)
        color = self.colors[colorIndex]

        # Ensure the color values are integers
        color = tuple([int(c) for c in color])

        frame[top:bottom + 1, left:right + 1][mask] = ([0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

        mask = mask.astype(np.uint8)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, 3, cv.LINE_8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mask-RCNN object detection and segmentation')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument("--device", default="cpu", help="Inference device (cpu/gpu)")
    args = parser.parse_args()

    mask_rcnn = MaskRCNN(args)
    mask_rcnn.run()
