import numpy as np

def detect_clicks(fdr_curve, t, threshold):
  
    above = fdr_curve > threshold

    diff = np.diff(above.astype(int))
    region_starts = np.where(diff == 1)[0] + 1  
    region_ends = np.where(diff == -1)[0] + 1

    if above[0]:
        region_starts = np.insert(region_starts, 0, 0)
    if above[-1]:
        region_ends = np.append(region_ends, len(fdr_curve))

    detections = []
    for start, end in zip(region_starts, region_ends):
        region_slice = slice(start, end)
        local_max_idx = np.argmax(fdr_curve[region_slice]) + start

        detections.append({
            "time": t[local_max_idx],
            "fdr_value": fdr_curve[local_max_idx],
            "start_time": t[start],
            "end_time": t[end - 1],
        })

    return detections