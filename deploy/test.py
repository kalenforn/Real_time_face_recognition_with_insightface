import cv2
import argparse

from model import Recognition

def get_args():
    
    parser = argparse.ArgumentParser(description="argument of face recognition.")
    #parser.add_argument('pattern', choices=['camera', 'image', 'video'], help="the pattern of this programe work. camera: using camera to detect, image: detecting an image, video: detecting a vido")
    parser.add_argument("--save-path", default="../datasets/output/", type=str, help="save path of the result")
    parser.add_argument('--database-path', default='../datasets/database/', help='database path', type=str)
    parser.add_argument('--image-size', default='112,112', help='corp image size')
    parser.add_argument('--model', default='../models/model-y1-test2/model,0000', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    
    sgroup = parser.add_argument_group("select operation")
    sgroup.add_argument('--save','-s', action='store_true', help="whether save teh result")
    sgroup.add_argument('--npy','-n', action='store_true', help="choose to load npy file or image file")

    igroup = parser.add_argument_group("operation of image detect")
    igroup.add_argument("--image-path", default="../dataset/test_image/test.jpg", type=str, help="the image path")

    vgroup = parser.add_argument_group("operation of video detect")
    vgroup.add_argument("--video-path", default="../dataset/test_video/test.mp4", type=str, help="the video path")

    ngroup = parser.add_argument_group("operation of npy file")
    ngroup.add_argument("--npy-datas", default="../datasets/npy/datas.npy", type=str, help="path of datas.npy")
    ngroup.add_argument("--npy-labels", default="../datasets/npy/labels.npy", type=str, help="path of labels.npy")

    args = parser.parse_args()

    return args

def main(args):
    model = Recognition(args, (args.npy_datas, args.npy_labels) if args.npy else (args.image_path), npy=args.npy)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        frame = cap.read()[1]

        frame = model.detect_frame(frame)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_args()
    main(args)


