import numpy as np
import SimpleITK as sitk

from onix.objects import OnixMask, OnixSpectra, OnixVolume


def calculate_metrics(analysis_1, analysis_2, map_name, regions):
    """ """
    map_1 = analysis_1.maps(map_name)
    map_2 = analysis_2.maps(map_name)

    if map_1 is None or map_2 is None:
        # TODO log warning / error
        return None

    # Compute SSIM in SI coordinates
    _, ssim_si = structural_similarity(
        map_1.array,
        map_2.array,
        win_size=5,
        data_range=1,
        full=True,
    )
    
    # Compute SSIM in MRI coordinates
    _, ssim_mri = structural_similarity(
        sitk.GetArrayFromImage(map_1.image),
        sitk.GetArrayFromImage(map_2.image),
        win_size=21,
        data_range=1,
        full=True,
    )
    
    # Compute the metrics for each region
    for region_name in regions:

        region_1 = analysis_1.mask(region_name)
        region_2 = analysis_2.mask(region_name)

        if region_1 is None and region_2 is None:
            # TODO log warning / error
            continue

        region = region_1 or region_2

        # Initialize first columns of current row
        row = {
            "dataset_1": a,
            "dataset_2": b,
            "map": map_name,
            "region": region_name,
            "region_dataset": r,
        }

        # Calculate OLS R^2 in SI coordinates
        map_1_masked = map_1.array[np.where(region.array)]
        map_2_masked = map_2.array[np.where(region.array)]
        mod = LinearRegression().fit(map_1_masked.reshape(-1,1), map_2_masked)
        metric_df.append(
            row | {
                "image_dimension": "SI",
                "metric": "OLS_r2",
                "value": mod.score(map_1_masked.reshape(-1,1), map_2_masked),
            }
        )

        # Calculate OLS R^2 in MRI coordinates
        map_1_masked = sitk.GetArrayFromImage(map_1.image)[np.where(sitk.GetArrayFromImage(region.image))]
        map_2_masked = sitk.GetArrayFromImage(map_2.image)[np.where(sitk.GetArrayFromImage(region.image))]
        mod = LinearRegression().fit(map_1_masked.reshape(-1,1), map_2_masked)
        metric_df.append(
            row | {
                "image_dimension": "MRI",
                "metric": "OLS_r2",
                "value": mod.score(map_1_masked.reshape(-1,1), map_2_masked),
            }
        )

        # Compute SSIM in SI coordinates
        metric_df.append(
            row | {
                "image_dimension": "SI",
                "metric": "SSIM",
                "value": np.mean(ssim_si[np.where(region.array)]),
            }
        )
        
        # Compute SSIM in MRI coordinates
        metric_df.append(
            row | {
                "image_dimension": "MRI",
                "metric": "SSIM",
                "value": np.mean(ssim_mri[np.where(sitk.GetArrayFromImage(region.image))]),
            }
        )

def batch_process(
    sessions: dict[str, OnixDataset],
    analysis_pairs: list[tuple[str, str]],
    maps: list[str],
    masks: list[str],
    regions: list[str],
    save_path: Path | str,
):
    """ """
    save_path = Path(save_path)
    results = []

    os.makedirs(save_path, exist_ok=False)

    for session_name, session in sessions.items():

        session_path = save_path / session_name
        os.makedirs(session_path, exist_ok=False)

        for a1, a2 in analysis_pairs:

            path = session_path / f"{analysis_1}-vs-{analysis_2}.csv"
            metric_df = []

            analysis_1 = session.mrsi_analyses[a1]
            analysis_2 = session.mrsi_analyses[a2]
            
            # Compute the metrics for each map
            for map_name in maps:

                metric_df += calculate_metrics(
                    analysis_1, 
                    analysis_2, 
                    map_name,
                    regions,
                )

            metric_df = pd.DataFrame(metric_df)
            metric_df.to_csv(path, index=False)
