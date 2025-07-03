import torch
import numpy as np
from tqdm import tqdm
import gc
from ..trainer import predict


def evaluate(config,
                  model,
                  query_loader,
                  gallery_loader,
                  ranks=[1, 5, 10],
                  step_size=1000,
                  cleanup=True):
    
    
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
    
    # --- START OF CODE TO ADD FOR SUBMISSION FILE GENERATION ---

    img_features_query_cpu = img_features_query.cpu().float()
    img_features_gallery_cpu = img_features_gallery.cpu().float()

    # Calculate similarity matrix (assuming features are L2-normalized, dot product is cosine similarity)
    similarity_matrix = torch.mm(img_features_query_cpu, img_features_gallery_cpu.T)

    k_predictions = 10

    topk_values, topk_indices = torch.topk(similarity_matrix, k=k_predictions, dim=1)

    # Define the output directory for your submission file
    # This will create a 'submission_results' folder in your main GeoAIM2_AI4E2025 directory
    submission_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../submission_results')
    os.makedirs(submission_output_dir, exist_ok=True)
    submission_filepath = os.path.join(submission_output_dir, 'answer.txt')

    with open(submission_filepath, 'w') as f:
        for i, query_id_full_path in enumerate(ids_query):
            # Extract base ID (e.g., 'image-01.jpeg' -> 'image-01')
            query_base_id = os.path.splitext(os.path.basename(query_id_full_path))[0]

            ranked_gallery_base_ids = []
            for j in topk_indices[i]:
                # Extract base ID from gallery (e.g., '0839.jpg' -> '0839')
                gallery_id_full_path = ids_gallery[j.item()]
                gallery_base_id = os.path.splitext(os.path.basename(gallery_id_full_path))[0]
                ranked_gallery_base_ids.append(gallery_base_id)

            # Format the line: query_id space-separated ranked_gallery_ids
            line = f"{query_base_id} {' '.join(ranked_gallery_base_ids)}\n"
            f.write(line)

    print(f"Submission file 'answer.txt' created successfully at: {submission_filepath}")
    print("--- END OF SUBMISSION FILE GENERATION ---")
    # --- END OF CODE TO ADD FOR SUBMISSION FILE GENERATION ---

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()
    
    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    
    AP = ap/len(ids_query)*100
    
    CMC = CMC.float()
    CMC = CMC/len(ids_query) #average CMC
    
    # top 1%
    top1 = round(len(ids_gallery)*0.01)
    
    string = []
             
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
    string.append('AP: {:.4f}'.format(AP))             
        
    print(' - '.join(string)) 
    
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        #torch.cuda.empty_cache()
    
    return CMC[0]


def eval_query(qf,ql,gf,gl):

    score = gf @ qf.unsqueeze(-1)
    
    score = score.squeeze().cpu().numpy()
 
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]    

    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl==-1)
    
    
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc




