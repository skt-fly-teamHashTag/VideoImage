import cv2
import numpy as np
import torch
##visualize##
import matplotlib.pyplot as plt  

from helpers import init_helper, vsumm_helper, bbox_helper, video_helper
from modules.model_zoo import get_model


def main():
    args = init_helper.get_arguments()

    # load model
    print('Loading DSNet model ...')
    model = get_model(args.model, **vars(args))
    print("---1---")
    model = model.eval().to(args.device)
    print("---2---")
    state_dict = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    print("---3---")
    model.load_state_dict(state_dict)
    print("---4---")

    # load video
    print('Preprocessing source video ...')
    video_proc = video_helper.VideoPreprocessor(args.sample_rate)
    print("---5---")
    n_frames, seq, cps, nfps, picks = video_proc.run(args.source) 
    seq_len = len(seq)
    print("---6---")

    print('Predicting summary ...')
    with torch.no_grad():
        seq_torch = torch.from_numpy(seq).unsqueeze(0).to(args.device)

        pred_cls, pred_bboxes = model.predict(seq_torch)

        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
        
        print(f"before soft-nms, len pred_bboxes: {len(pred_bboxes)}")
        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, args.nms_thresh) # th보다 큰 score을 가지는 bbox, score 쌍을 리턴
        # pred_cls, pred_bboxes = bbox_helper.soft_nms(pred_cls, pred_bboxes, args.nms_thresh) 
        print(f"after soft-nms, len pred_bboxes: {len(pred_bboxes)}")
        pred_summ = vsumm_helper.bbox2summary(
            seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks) #apply knapsack
        
        ##visualize##
        print("prediction summary len: ", len(pred_summ), '#of true: ', pred_summ.astype('int').sum())
        plt.plot(pred_summ, 'go', markersize=1)
        figure_y = [0.1, 0.15]
        # for i in range(len(cps)-1):
        #     plt.axvline(cps[i][1], 0, 1, color='lightgray', linestyle='solid', linewidth=1) 
        #     plt.text(cps[i][1], figure_y[i%2],
        #             str(int(((1/fps)*cps[i][1]))//60)+':'+str(int(((1/fps)*cps[i][1]))%60), 
        #             color='gray',
        #             fontsize = 7,
        #             horizontalalignment='center',
        #             verticalalignment='bottom')
        plt.savefig('../output/ours/soyoung0_ab_sr10_kanplimit0p3_softnms0p5_sum0.png') 
        print("predicted figure saved")
    
    print('Writing summary video ...')

    
    # load original video
    cap = cv2.VideoCapture(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create summary video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height))

    frame_idx = 0 
    while True:
        # ret, frame = cap.read()
        # ret = cap.grab() 
        ret = cap.grab()
        ret, frame = cap.retrieve()

        if not ret:
            break

        if pred_summ[frame_idx]:
            out.write(frame)

        # cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)+1)
        frame_idx += 1

    out.release()
    cap.release()
    


if __name__ == '__main__':
    main()
