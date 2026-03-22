import torch
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getcwd(), "moment_detr"))
from run_on_video.model_utils import build_inference_model
from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.run import *
from data_preparation import extract_and_combine

#------------------------------------------------------

class myMomentDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 2  # seconds
        self.device = device
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained Moment-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def extract_clip_features(self, video_path):
        """
        Args:
            video_path: str, path to the video file
        Returns:
            tensor, video features
        """
        # extract video features from video
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        return video_feats

    def preprocess(self, video_feats, query_list):
        """
        Args:
            video_feats: tensor, video features
            query_list: list of str queries
        Returns:
            dict, preprocessed model input and
            int, number of frames corresponding to the video features
        """
        # construct model inputs
        # add tef
        n_frames = len(video_feats)
        n_query = len(query_list)
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        video_feats = torch.cat([video_feats, tef], dim=1)
        assert n_frames <= 75, "The positional embedding of this pretrained MomentDETR only support video up " \
                               "to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )
        return model_inputs, n_frames

    @torch.no_grad()
    def process(self, model_inputs, n_frames, query_list):
        """
        Args:
            model_inputs: dict, with model inputs
            n_frames: int, number of frames being processed
            query_list: list of str queries
        Returns:
            list, predictions
        """
        # decode outputs
        outputs = self.model(**model_inputs)
        # #moment_queries refers to the positional embeddings in MomentDETR's decoder, not the input text query
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #moment_queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        # compose predictions
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query=query_list[idx],  # str
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,  # List([st(float), ed(float), score(float)])
                pred_saliency_scores=saliency_scores[idx]  # List(float), len==n_frames, scores for each frame
            )
            predictions.append(cur_query_pred)
        return predictions

    def localize_moment(self, video_path, query_list):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        Returns:
            list of dict: outputs
        """
        feats = self.extract_clip_features(video_path)
        inputs, n_frames = self.preprocess(feats, query_list)
        outputs = self.process(inputs, n_frames, query_list)
        return outputs

    def pretty_print_pred(self, predictions, query_text_list, relevant_threshold=0.0):
        """
        Args:
            predictions: dict, with model inputs
            query_text_list: list of str queries
            relevant_threshold: float, predictde relevant probability threshold
        Returns:
            list, predictions
        """
        predictions_filtered = [{
                'pred_relevant_windows': [m for m in p['pred_relevant_windows'] if m[2] > relevant_threshold]
            }
            for p in predictions]
        # - Print pred
        for idx, query_data in enumerate(query_text_list):
            print("-"*30 + f"idx{idx}")
            print(f">> query: {query_data}")
            print(f">> Predicted moments ([start_in_seconds, end_in_seconds, score]): "
                f"{predictions_filtered[idx]['pred_relevant_windows']}")
            print(f">> Predicted saliency scores (for all 2-sec clip): "
                f"{predictions[idx]['pred_saliency_scores']}")

#------------------------------------------------------

def infer_long_video(video_path, query_list, relevant_threshold):
    """
    Args:
        video_path: str, path to the video file
        query_list: list of str queries
    Returns:
        list, saliency scores
    """
    # - Load the model
    print("\n - Init myMomentDETRPredictor")
    moment_detr_predictor = myMomentDETRPredictor(
            ckpt_path=ckpt_path,
            clip_model_name_or_path=clip_model_name_or_path,
            device=device)

    # - Extract Clip features of the full video
    print("\n - Extract Clip features")
    video_feats =  moment_detr_predictor.extract_clip_features(video_path)
    n_total_frames = len(video_feats)
    print(f"n_total_frames={n_total_frames}")

    # - split the data into sliding windows of 75 frames
    step = 75
    windows_size = 75
    full_saliency = [np.zeros(n_total_frames) for _ in query_list]
    counts = [np.zeros(n_total_frames) for _ in query_list]
    predicted_moments = [[] for _ in query_list]

    for start_id in range (0, n_total_frames, step):
        end_id = min(start_id + windows_size, n_total_frames)
        print(f"\n - Predicting clip frames [{start_id}:{end_id}]")
        video_feats_segment = video_feats[start_id:end_id]
        inputs_segment, n_frames_seg = moment_detr_predictor.preprocess(video_feats_segment, query_list)

        # predict
        outputs_segment = moment_detr_predictor.process(inputs_segment, n_frames_seg, query_list)
        moment_detr_predictor.pretty_print_pred(outputs_segment, query_list, relevant_threshold)

        # save results for segment
        for i,_ in enumerate(query_list):
            full_saliency[i][start_id:end_id] += np.array(outputs_segment[i]["pred_saliency_scores"]) #*5
            counts[i][start_id:end_id] += 1
            # add relative start and end
            for m in outputs_segment[i]["pred_relevant_windows"]:
                if m[2] > relevant_threshold:
                    predicted_moments[i].append([m[0]+start_id*2, m[1]+start_id*2])

    final_saliency_scores = full_saliency / np.maximum(counts, 1)
    return final_saliency_scores, predicted_moments


def plot_saliency_scores(scores, query_list, threshold=3.5, output_path="saliency_plot"):

    num_clips = len(scores[0])
    time_axis = np.arange(num_clips) * 2  # Convert index to seconds

    for i, q in enumerate(query_list):
        plt.figure(figsize=(12, 5))

        # Plot the scores
        plt.plot(time_axis, scores[i], color='blue', linewidth=2, label='Saliency Score')

        # Add a horizontal line for the threshold
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')

        # Formatting
        plt.title('Moment-DETR Saliency Score for Q ='+q, fontsize=14)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Score (0-5)', fontsize=12)
        plt.ylim(-1, 5.5)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.legend(loc='upper right')

        plt.tight_layout()
        out = output_path+str(i)+".png"
        plt.savefig(out)
        print(f"Plot successfully saved to {out}")


def plot_predictions(saliency_scores, predicted_moments, query_list, output_path = "prediction_plot"):
    """
    Args:
        saliency_scores: list, scores for each frame per query
        predicted_moments: list of (start , end) moments predictions in s
        query_list: list of str queries
        output_path: str, path to plot image
    """
    # time axis
    video_duration = 2 * len(saliency_scores[0]) # 1 frame is 2s
    time_axis = np.linspace(0, video_duration, len(saliency_scores[0]))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22']

    for i, q in enumerate(query_list):
        plt.figure(figsize=(12, 5))

        # - plot saliency scores
        plt.plot(time_axis, saliency_scores[i], color='#2c3e50', linewidth=1.5, label='Saliency Score')
        plt.fill_between(time_axis, saliency_scores[i], -1, color='#3498db', alpha=0.2)
        plt.axhline(0, color='#c0392b', linestyle='--', linewidth=1)

        # - plot predicted moments
        for j, (start, end) in enumerate(predicted_moments[i]):
            current_color = colors[j % len(colors)]
            plt.axvspan(start, end, color=current_color, alpha=0.3, label="")
            plt.axvline(start, color=current_color, linestyle='--', linewidth=1)
            plt.axvline(end, color=current_color, linestyle='--', linewidth=1)
            plt.axvline((start + end) / 2, color=current_color, linestyle='-', linewidth=1)

        plt.title("Saliency Scores & Predicted Spans- Q ="+ q, fontsize=14)
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Score (-1 to 1)", fontsize=12)
        plt.ylim(-1.1, 1.1)
        plt.xlim(0, video_duration)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.legend(loc='lower right')

        plt.tight_layout()
        out = output_path+str(i)+".png"
        plt.savefig(out)
        print(f"Plot successfully saved to {out}")

#------------------------------------------------------

if __name__ == "__main__":

    # -----------------------
    # - inputs
    device = 'cuda'
    ckpt_path ="./moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt"
    clip_model_name_or_path = "ViT-B/32"
    video_path = "../data/Janja_Garnbret.mp4"
    # video_path = "../data/video_segments_350/segment_1.mp4"
    query_text_list = ["a person jumping", "a person climbing and falling", "a crowd cheering"]
    relevant_threshold = 0.95
    # -----------------------

    # - infer
    saliency_scores, predicted_moments = infer_long_video(video_path, query_text_list, relevant_threshold)

    # - plot
    plot_predictions(saliency_scores, predicted_moments, query_text_list, output_path="../data/Janja_Garnbret_plot")

    # - generate videos
    for i, q in enumerate(query_text_list):
        extract_and_combine(video_path, predicted_moments[i], "../data/retrieved_highlights_"+str(i)+".mp4", q)