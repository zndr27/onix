from Onix.objects import OnixObject, OnixVolume

class OnixMetrics(OnixObject):
    """ """

    fitt_ds: OnixDataset
    nnfit_ds: OnixDataset

    ref: OnixVolume

    nawm: itk.Image

    ratio_table: dash_table.DataTable
    map_table: dash_table.DataTable

    fitt_cho_naa_mask: None | OnixVolume = None
    nnfit_cho_naa_mask: None | OnixVolume = None
    
    fitt_cho_naa_mask_1connect: None | OnixVolume = None
    nnfit_cho_naa_mask_1connect: None | OnixVolume = None

    def __init__(
        self,
        fitt_ds: OnixDataset,
        nnfit_ds: OnixDataset,
        t1_tx,
        ref_tx,
        brain_mask: OnixVolume,
        qmap: OnixVolume,
        save_outputs=True,
        overwrite_ccb=True,
    ):
        super().__init__()

        self.fitt_ds = fitt_ds
        self.nnfit_ds = nnfit_ds

        self.t1_tx = t1_tx
        self.ref_tx = ref_tx

        self.brain_mask = brain_mask
        self.qmap = qmap

        # TODO >>>
        self.hqmap = fitt_ds.hqmap
        self.vhqmap = fitt_ds.vhqmap
        self.nnqmap = fitt_ds.nnqmap
        self.nnhqmap = fitt_ds.nnhqmap
        self.nnvhqmap = fitt_ds.nnvhqmap
        # TODO <<<

        self.nawm = self.get_nawm()

        onix_path = self.fitt_ds.onix_path
        self.onix_path = onix_path

        if save_outputs:

            print("GENERATING METRICS")

            fitt_cho_naa = self.fitt_ds.load_sinorm_map("CHO/NAA")
            fitt_cho_naa = fitt_cho_naa.align(self.fitt_ds.t1)

            nnfit_cho_naa = self.nnfit_ds.load_nnfit_map("cho/naa ratio")
            nnfit_cho_naa = nnfit_cho_naa.register(self.fitt_ds.ref, self.ref_tx)
            nnfit_cho_naa = nnfit_cho_naa.align(self.fitt_ds.t1)

            self.fitt_cho_naa = fitt_cho_naa
            self.nnfit_cho_naa = nnfit_cho_naa

            with open(self.onix_path / 'fitt_cho_naa.pkl', 'wb') as f:
                pickle.dump(self.fitt_cho_naa, f, pickle.HIGHEST_PROTOCOL)
            with open(self.onix_path / 'nnfit_cho_naa.pkl', 'wb') as f:
                pickle.dump(self.nnfit_cho_naa, f, pickle.HIGHEST_PROTOCOL)
            
            yeet = time.time()
            self.map_df, self.map_df_t1 = self.get_map_df()
            self.map_table = self.get_table(self.map_df)
            self.map_table_t1 = self.get_table(self.map_df_t1)
            print("map time", yeet - time.time())
            
            yeet = time.time()
            self.ratio_df = self.get_cho_naa_df(overwrite_ccb=overwrite_ccb)
            self.ratio_table = self.get_table(self.ratio_df)
            #self.ratio_table = self.get_cho_naa_table(overwrite_ccb=overwrite_ccb)
            print('cho_naa table time', yeet - time.time())
            ###self.ratio_table = self.map_table

            with open(onix_path / "map_df.pkl", 'wb') as f:
                pickle.dump(self.map_df, f, pickle.HIGHEST_PROTOCOL)
            with open(onix_path / "map_df_t1.pkl", 'wb') as f:
                pickle.dump(self.map_df_t1, f, pickle.HIGHEST_PROTOCOL)
            with open(onix_path / "map_table.pkl", 'wb') as f:
                pickle.dump(self.map_table, f, pickle.HIGHEST_PROTOCOL)
            with open(onix_path / "map_table_t1.pkl", 'wb') as f:
                pickle.dump(self.map_table_t1, f, pickle.HIGHEST_PROTOCOL)
            with open(onix_path / "ratio_df.pkl", 'wb') as f:
                pickle.dump(self.ratio_df, f, pickle.HIGHEST_PROTOCOL)
            with open(onix_path / "ratio_table.pkl", 'wb') as f:
                pickle.dump(self.ratio_table, f, pickle.HIGHEST_PROTOCOL)

            with open(self.onix_path / 'fitt_cho_naa_mask.pkl', 'wb') as f:
                pickle.dump(self.fitt_cho_naa_mask, f, pickle.HIGHEST_PROTOCOL)
            with open(self.onix_path / 'nnfit_cho_naa_mask.pkl', 'wb') as f:
                pickle.dump(self.nnfit_cho_naa_mask, f, pickle.HIGHEST_PROTOCOL)
            with open(self.onix_path / 'fitt_cho_naa_mask_1connect.pkl', 'wb') as f:
                pickle.dump(self.fitt_cho_naa_mask_1connect, f, pickle.HIGHEST_PROTOCOL)
            with open(self.onix_path / 'nnfit_cho_naa_mask_1connect.pkl', 'wb') as f:
                pickle.dump(self.nnfit_cho_naa_mask_1connect, f, pickle.HIGHEST_PROTOCOL)
            with open(self.onix_path / 'fitt_cho_naa_mask_2connect.pkl', 'wb') as f:
                pickle.dump(self.fitt_cho_naa_mask_2connect, f, pickle.HIGHEST_PROTOCOL)
            with open(self.onix_path / 'nnfit_cho_naa_mask_2connect.pkl', 'wb') as f:
                pickle.dump(self.nnfit_cho_naa_mask_2connect, f, pickle.HIGHEST_PROTOCOL)

            with open(self.onix_path / 'ccb_mask.pkl', 'wb') as f:
                pickle.dump(self.ccb_mask, f, pickle.HIGHEST_PROTOCOL)

        else:
            
            print("LOADING METRICS")

            with open(self.onix_path / 'fitt_cho_naa.pkl', 'rb') as f:
                self.fitt_cho_naa = pickle.load(f)
            with open(self.onix_path / 'nnfit_cho_naa.pkl', 'rb') as f:
                self.nnfit_cho_naa = pickle.load(f)

            with open(onix_path / "map_df.pkl", 'rb') as f:
                self.map_df = pickle.load(f)
            with open(onix_path / "map_df_t1.pkl", 'rb') as f:
                self.map_df_t1 = pickle.load(f)
            with open(onix_path / "map_table.pkl", 'rb') as f:
                self.map_table = pickle.load(f)
            with open(onix_path / "map_table_t1.pkl", 'rb') as f:
                self.map_table_t1 = pickle.load(f)
            with open(onix_path / "ratio_df.pkl", 'rb') as f:
                self.ratio_df = pickle.load(f)
            with open(onix_path / "ratio_table.pkl", 'rb') as f:
                self.ratio_table = pickle.load(f)

            with open(self.onix_path / 'fitt_cho_naa_mask.pkl', 'rb') as f:
                self.fitt_cho_naa_mask = pickle.load(f)
            with open(self.onix_path / 'nnfit_cho_naa_mask.pkl', 'rb') as f:
                self.nnfit_cho_naa_mask = pickle.load(f)
            with open(self.onix_path / 'fitt_cho_naa_mask_1connect.pkl', 'rb') as f:
                self.fitt_cho_naa_mask_1connect = pickle.load(f)
            with open(self.onix_path / 'nnfit_cho_naa_mask_1connect.pkl', 'rb') as f:
                self.nnfit_cho_naa_mask_1connect = pickle.load(f)
            with open(self.onix_path / 'fitt_cho_naa_mask_2connect.pkl', 'rb') as f:
                self.fitt_cho_naa_mask_2connect = pickle.load(f)
            with open(self.onix_path / 'nnfit_cho_naa_mask_2connect.pkl', 'rb') as f:
                self.nnfit_cho_naa_mask_2connect = pickle.load(f)
            
            with open(self.onix_path / 'ccb_mask.pkl', 'rb') as f:
                self.ccb_mask = pickle.load(f)

    def save_metrics(self, save_path: Path):
        """ """
        # Append subject id and date to save_path
        save_path = Path(save_path)
        save_path = save_path / self.fitt_ds.study.subject_id 
        save_path = save_path / self.fitt_ds.study.date.replace('/', '_')
        os.makedirs(save_path, exist_ok=True)

        # Save map_df using pandas to csv
        self.map_df.to_csv(save_path / "map_df.csv", index=False)
        self.map_df_t1.to_csv(save_path / "map_df_t1.csv", index=False)

        # Save ratio_df using pandas to csv
        self.ratio_df.to_csv(save_path / "ratio_df.csv", index=False)

    def get_nawm(self):
        """ """
        fitt_nawm = self.fitt_ds.study.subject_path / "mri"
        fitt_nawm = fitt_nawm / (
            "*" + self.fitt_ds.study.param("Study_Date").replace("/", "_") + "*nawm.img"
        )
        fitt_nawm = glob.glob(str(fitt_nawm))
        if fitt_nawm != []:
            return fitt_nawm[0]

        nnfit_nawm = self.nnfit_ds.study.subject_path / "mri"
        nnfit_nawm = nnfit_nawm / (
            "*"
            + self.nnfit_ds.study.param("Study_Date").replace("/", "_")
            + "*nawm.img"
        )
        nnfit_nawm = glob.glob(str(nnfit_nawm))
        if nnfit_nawm != []:
            return nnfit_nawm[0]

        raise Exception(f"No nawm file available")

    def nawm_threshold(self, image, threshold=2.0, avg=None):
        """ """
        array = np.copy(itk.GetArrayFromImage(image))

        nawm_mask = np.fromfile(self.nawm, dtype=np.uint8).reshape(array.shape)

        if avg is None:
            nawm_avg = np.sum(array.flatten()[nawm_mask.flatten() > 0]) / np.sum(
                nawm_mask.flatten() > 0
            )
        else:
            nawm_avg = avg

        threshold_filter = itk.BinaryThresholdImageFilter[
            itk.Image[itk.F, 3], itk.Image[itk.UC, 3]
        ].New()
        threshold_filter.SetInput(image)
        threshold_filter.SetUpperThreshold(threshold * nawm_avg)
        threshold_filter.SetInsideValue(0)
        threshold_filter.SetOutsideValue(1)
        output = threshold_filter.GetOutput()

        return OnixVolume(itk.GetArrayFromImage(output), output, is_mask=True), nawm_avg

    def compare_map_correlation(self, a1: np.ndarray, a2: np.ndarray):
        """ """
        assert a1.shape == a2.shape

        brain_mask_arg = np.where(self.brain_mask.array.flatten() == 1)[0]
        qmap_arg = np.where(self.qmap.array.flatten() == 4)[0]
        hqmap_arg = np.where(self.hqmap.array.flatten() == 1)[0]
        vhqmap_arg = np.where(self.vhqmap.array.flatten() == 1)[0]
        nnqmap_arg = np.where(self.nnqmap.array.flatten() == 1)[0]
        nnhqmap_arg = np.where(self.nnhqmap.array.flatten() == 1)[0]
        nnvhqmap_arg = np.where(self.nnvhqmap.array.flatten() == 1)[0]
        nnvhq_arg = np.intersect1d(nnvhqmap_arg, vhqmap_arg)

        data = {}

        for argname, arg in {
            'brain_mask': brain_mask_arg,
            'qmap': qmap_arg,
            'hqmap': hqmap_arg,
            'vhqmap': vhqmap_arg,
            'nnqmap': nnqmap_arg,
            'nnhqmap': nnhqmap_arg,
            'nnvhqmap': nnvhqmap_arg,
            'nnvhq': nnvhq_arg,
        }.items():
            mod_ = LinearRegression().fit(a1.flatten()[arg].reshape(-1,1), a2.flatten()[arg])
            data |= {
                f"_R2_{argname}": mod_.score(a1.flatten()[arg].reshape(-1,1), a2.flatten()[arg]),
                f"_coef_{argname}": mod_.coef_,
                f"_intercept_{argname}": mod_.intercept_,
            }
            result = linregress(a1.flatten()[arg], a2.flatten()[arg])
            data |= {
                f"slope_{argname}": result.slope,
                f"intercept_{argname}": result.intercept,
                f"rvalue_{argname}": result.rvalue,
                f"pvalue_{argname}": result.pvalue,
                f"stderr_{argname}": result.stderr,
                f"intercept_stderr_{argname}": result.intercept_stderr,
            }

        return data

    def compare_map_3d(self, a1: np.ndarray, a2: np.ndarray):
        """ """
        assert a1.shape == a2.shape
        mssim, S = structural_similarity(a1, a2, data_range=1, full=True)

        brain_mask_arg = np.where(self.brain_mask.array.flatten() == 1)[0]
        qmap_arg = np.where(self.qmap.array.flatten() == 4)[0]
        hqmap_arg = np.where(self.hqmap.array.flatten() == 1)[0]
        vhqmap_arg = np.where(self.vhqmap.array.flatten() == 1)[0]
        nnqmap_arg = np.where(self.nnqmap.array.flatten() == 1)[0]
        nnhqmap_arg = np.where(self.nnhqmap.array.flatten() == 1)[0]
        nnvhqmap_arg = np.where(self.nnvhqmap.array.flatten() == 1)[0]
        nnvhq_arg = np.intersect1d(nnvhqmap_arg, vhqmap_arg)

        mssim_brain = np.sum(S.flatten()[brain_mask_arg]) / len(brain_mask_arg)
        mssim_qmap = np.sum(S.flatten()[qmap_arg]) / len(qmap_arg)
        mssim_hqmap = np.sum(S.flatten()[hqmap_arg]) / len(hqmap_arg)
        mssim_vhqmap = np.sum(S.flatten()[vhqmap_arg]) / len(vhqmap_arg)
        mssim_nnqmap = np.sum(S.flatten()[nnqmap_arg]) / len(nnqmap_arg)
        mssim_nnhqmap = np.sum(S.flatten()[nnhqmap_arg]) / len(nnhqmap_arg)
        mssim_nnvhqmap = np.sum(S.flatten()[nnvhqmap_arg]) / len(nnvhqmap_arg)
        mssim_nnvhq = np.sum(S.flatten()[nnvhq_arg]) / len(nnvhq_arg)

        mean_ssim_brain = np.mean(S.flatten()[brain_mask_arg])
        mean_ssim_qmap = np.mean(S.flatten()[qmap_arg])
        mean_ssim_hqmap = np.mean(S.flatten()[hqmap_arg])
        mean_ssim_vhqmap = np.mean(S.flatten()[vhqmap_arg])
        mean_ssim_nnqmap = np.mean(S.flatten()[nnqmap_arg])
        mean_ssim_nnhqmap = np.mean(S.flatten()[nnhqmap_arg])
        mean_ssim_nnvhqmap = np.mean(S.flatten()[nnvhqmap_arg])
        mean_ssim_nnvhq = np.mean(S.flatten()[nnvhq_arg])

        std_ssim_brain = np.std(S.flatten()[brain_mask_arg])
        std_ssim_qmap = np.std(S.flatten()[qmap_arg])
        std_ssim_hqmap = np.std(S.flatten()[hqmap_arg])
        std_ssim_vhqmap = np.std(S.flatten()[vhqmap_arg])
        std_ssim_nnqmap = np.std(S.flatten()[nnqmap_arg])
        std_ssim_nnhqmap = np.std(S.flatten()[nnhqmap_arg])
        std_ssim_nnvhqmap = np.std(S.flatten()[nnvhqmap_arg])
        std_ssim_nnvhq = np.std(S.flatten()[nnvhq_arg])

        return {
            "mse": mean_squared_error(a1, a2),
            #'nmi'            : normalized_mutual_information(a1, a2),
            #'nrmse'          : normalized_root_mse(a1, a2),
            #'mssim'          : mssim,
            "mssim brain": mssim_brain,
            "mssim qmap": mssim_qmap,
            "mssim hqmap": mssim_hqmap,
            "mssim vhqmap": mssim_vhqmap,
            "mssim nnqmap": mssim_nnqmap,
            "mssim nnhqmap": mssim_nnhqmap,
            "mssim nnvhqmap": mssim_nnvhqmap,
            "mssim nnvhq": mssim_nnvhq,
            "mean_ssim_brain": mean_ssim_brain,
            "mean_ssim_qmap": mean_ssim_qmap,
            "mean_ssim_hqmap": mean_ssim_hqmap,
            "mean_ssim_vhqmap": mean_ssim_vhqmap,
            "mean_ssim_nnqmap": mean_ssim_nnqmap,
            "mean_ssim_nnhqmap": mean_ssim_nnhqmap,
            "mean_ssim_nnvhqmap": mean_ssim_nnvhqmap,
            "mean_ssim_nnvhq": mean_ssim_nnvhq,
            "std_ssim_brain": std_ssim_brain,
            "std_ssim_qmap": std_ssim_qmap,
            "std_ssim_hqmap": std_ssim_hqmap,
            "std_ssim_vhqmap": std_ssim_vhqmap,
            "std_ssim_nnqmap": std_ssim_nnqmap,
            "std_ssim_nnhqmap": std_ssim_nnhqmap,
            "std_ssim_nnvhqmap": std_ssim_nnvhqmap,
            "std_ssim_nnvhq": std_ssim_nnvhq,
        }

    def compare_map_3d_itk(self, a1: np.ndarray, a2: np.ndarray):
        """ """
        assert a1.shape == a2.shape
        mssim, S = structural_similarity(a1, a2, win_size=21, data_range=1, full=True)

        brain_mask = itk.GetArrayFromImage(self.brain_mask.align(self.fitt_ds.t1).image)
        qmap = itk.GetArrayFromImage(self.qmap.align(self.fitt_ds.t1).image)
        hqmap = itk.GetArrayFromImage(self.hqmap.align(self.fitt_ds.t1).image)
        vhqmap = itk.GetArrayFromImage(self.vhqmap.align(self.fitt_ds.t1).image)
        nnqmap = itk.GetArrayFromImage(self.nnqmap.align(self.fitt_ds.t1).image)
        nnhqmap = itk.GetArrayFromImage(self.nnhqmap.align(self.fitt_ds.t1).image)
        nnvhqmap = itk.GetArrayFromImage(self.nnvhqmap.align(self.fitt_ds.t1).image)

        brain_mask_arg = np.where(brain_mask.flatten() > 0.5)[0]
        qmap_arg = np.where(qmap.flatten() > 3.5)[0]
        hqmap_arg = np.where(hqmap.flatten() > 0.5)[0]
        vhqmap_arg = np.where(vhqmap.flatten() > 0.5)[0]
        nnqmap_arg = np.where(nnqmap.flatten() > 0.5)[0]
        nnhqmap_arg = np.where(nnhqmap.flatten() > 0.5)[0]
        nnvhqmap_arg = np.where(nnvhqmap.flatten() > 0.5)[0]
        nnvhq_arg = np.intersect1d(nnvhqmap_arg, vhqmap_arg)

        mssim_brain = np.sum(S.flatten()[brain_mask_arg]) / len(brain_mask_arg)
        mssim_qmap = np.sum(S.flatten()[qmap_arg]) / len(qmap_arg)
        mssim_hqmap = np.sum(S.flatten()[hqmap_arg]) / len(hqmap_arg)
        mssim_vhqmap = np.sum(S.flatten()[vhqmap_arg]) / len(vhqmap_arg)
        mssim_nnqmap = np.sum(S.flatten()[nnqmap_arg]) / len(nnqmap_arg)
        mssim_nnhqmap = np.sum(S.flatten()[nnhqmap_arg]) / len(nnhqmap_arg)
        mssim_nnvhqmap = np.sum(S.flatten()[nnvhqmap_arg]) / len(nnvhqmap_arg)
        mssim_nnvhq = np.sum(S.flatten()[nnvhq_arg]) / len(nnvhq_arg)

        mean_ssim_brain = np.mean(S.flatten()[brain_mask_arg])
        mean_ssim_qmap = np.mean(S.flatten()[qmap_arg])
        mean_ssim_hqmap = np.mean(S.flatten()[hqmap_arg])
        mean_ssim_vhqmap = np.mean(S.flatten()[vhqmap_arg])
        mean_ssim_nnqmap = np.mean(S.flatten()[nnqmap_arg])
        mean_ssim_nnhqmap = np.mean(S.flatten()[nnhqmap_arg])
        mean_ssim_nnvhqmap = np.mean(S.flatten()[nnvhqmap_arg])
        mean_ssim_nnvhq = np.mean(S.flatten()[nnvhq_arg])

        std_ssim_brain = np.std(S.flatten()[brain_mask_arg])
        std_ssim_qmap = np.std(S.flatten()[qmap_arg])
        std_ssim_hqmap = np.std(S.flatten()[hqmap_arg])
        std_ssim_vhqmap = np.std(S.flatten()[vhqmap_arg])
        std_ssim_nnqmap = np.std(S.flatten()[nnqmap_arg])
        std_ssim_nnhqmap = np.std(S.flatten()[nnhqmap_arg])
        std_ssim_nnvhqmap = np.std(S.flatten()[nnvhqmap_arg])
        std_ssim_nnvhq = np.std(S.flatten()[nnvhq_arg])

        return {
            "mse": mean_squared_error(a1, a2),
            #'nmi'            : normalized_mutual_information(a1, a2),
            #'nrmse'          : normalized_root_mse(a1, a2),
            #'mssim'          : mssim,
            "mssim brain": mssim_brain,
            "mssim qmap": mssim_qmap,
            "mssim hqmap": mssim_hqmap,
            "mssim vhqmap": mssim_vhqmap,
            "mssim nnqmap": mssim_nnqmap,
            "mssim nnhqmap": mssim_nnhqmap,
            "mssim nnvhqmap": mssim_nnvhqmap,
            "mssim nnvhq": mssim_nnvhq,
            "mean_ssim_brain": mean_ssim_brain,
            "mean_ssim_qmap": mean_ssim_qmap,
            "mean_ssim_hqmap": mean_ssim_hqmap,
            "mean_ssim_vhqmap": mean_ssim_vhqmap,
            "mean_ssim_nnqmap": mean_ssim_nnqmap,
            "mean_ssim_nnhqmap": mean_ssim_nnhqmap,
            "mean_ssim_nnvhqmap": mean_ssim_nnvhqmap,
            "mean_ssim_nnvhq": mean_ssim_nnvhq,
            "std_ssim_brain": std_ssim_brain,
            "std_ssim_qmap": std_ssim_qmap,
            "std_ssim_hqmap": std_ssim_hqmap,
            "std_ssim_vhqmap": std_ssim_vhqmap,
            "std_ssim_nnqmap": std_ssim_nnqmap,
            "std_ssim_nnhqmap": std_ssim_nnhqmap,
            "std_ssim_nnvhqmap": std_ssim_nnvhqmap,
            "std_ssim_nnvhq": std_ssim_nnvhq,
        }

    def get_table(self, df: pd.DataFrame):
        """ """
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"id": c, "name": c} for c in df.columns],
            style_table={
                "overflowX": "auto",
            },
            style_header={
                "backgroundColor": "rgb(30, 30, 30)",
                "color": "white",
            },
            style_data={
                "backgroundColor": "rgb(50, 50, 50)",
                "color": "white",
            },
        )

    def preprocess_t1(self):
        """ Preprocess T1 image. """
        import os
        from pathlib import Path
        subject_path = self.fitt_ds.study.subject_path
        onix_path = subject_path / 'onix'
        
        subject_id = subject_path.name
        study_date = self.fitt_ds.study.date
        ### workfolder = f"preprocess-t1_{subject_id}_{study_date.replace('/','-')}"
        workfolder = "preprocess_t1"
        
        os.makedirs(onix_path, exist_ok=True)
        os.chdir(onix_path)
        
        import itk
        from nipype.interfaces.io import DataSink
        import nipype.interfaces.fsl as fsl
        import nipype.interfaces.freesurfer as freesurfer
        from nipype.interfaces.fsl.utils import RobustFOV
        from nipype.interfaces.utility import IdentityInterface
        import nipype.pipeline.engine as pe
        fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
        
        onix_path = onix_path / study_date.replace('/','_')
        img_file = onix_path / './struct.nii.gz'
        itk.imwrite(self.fitt_ds.t1.image, img_file)
        
        FSLDIR = Path(os.environ.get('FSLDIR'))
        mni_file = FSLDIR / './data/linearMNI/MNI152lin_T1_1mm_brain.nii.gz'
        cerebellum_file = FSLDIR / './data/atlases/Cerebellum/Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz'
        hocort_file = FSLDIR / './data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-1mm.nii.gz'
        hosub_file = FSLDIR / './data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz'
        
        preproc = pe.Workflow(
            name=workfolder, 
            base_dir=img_file.parent,
        )
        
        # TODO >>> Replacing BET with RobustFOV + Watershed Skull Strip

        ### bet = pe.Node(
        ###     interface=fsl.BET(
        ###         in_file=img_file,
        ###         mask=True,
        ###         skull=True,
        ###     ),
        ###     name='bet',
        ### )

        # Set size to 150mm vs default 170mm
        robust_fov = pe.Node(
            interface=RobustFOV(
                in_file=img_file,
                brainsize=150,
            ),
            name='robust_fov',
        )

        watershed = pe.Node(
            interface=freesurfer.WatershedSkullStrip(),
            name='watershed',
        )
        preproc.connect(robust_fov, 'out_roi', watershed, 'in_file')

        mri_convert = pe.Node(
            interface=freesurfer.MRIConvert(
                out_type='niigz',
            ),
            name='mri_convert',
        )
        preproc.connect(watershed, 'out_file', mri_convert, 'in_file')

        # TODO <<<

        fast = pe.Node(
            interface=fsl.FAST(
                output_biascorrected=True,
                output_biasfield=True,
                probability_maps=True,
            ),
            name='fast',
        )
        preproc.connect(mri_convert, 'out_file', fast, 'in_files')

        fast_sinker = pe.Node(
            interface = DataSink(
                base_directory = str(img_file.parent),
            ),
            name='fast_sinker',
        )
        preproc.connect(fast, 'bias_field', fast_sinker, 'fast_sinker.@bias_field')
        preproc.connect(fast, 'mixeltype', fast_sinker, 'fast_sinker.@mixeltype')
        preproc.connect(fast, 'partial_volume_files', fast_sinker, 'fast_sinker.@partial_volume_files')
        preproc.connect(fast, 'partial_volume_map', fast_sinker, 'fast_sinker.@partial_volume_map')
        preproc.connect(fast, 'probability_maps', fast_sinker, 'fast_sinker.@probability_maps')
        preproc.connect(fast, 'restored_image', fast_sinker, 'fast_sinker.@restored_image')
        preproc.connect(fast, 'tissue_class_files', fast_sinker, 'fast_sinker.@tissue_class_files')
        preproc.connect(fast, 'tissue_class_map', fast_sinker, 'fast_sinker.@tissue_class_map')
        
        flirt = pe.Node(
            interface=fsl.FLIRT(
                reference=mni_file,
            ),
            name='flirt',
        )
        preproc.connect(fast, 'restored_image', flirt, 'in_file')

        fnirt = pe.Node(
            interface=fsl.FNIRT(
                ref_file=mni_file,
                subsampling_scheme=[8,4,2,2],
                field_file=True,
                fieldcoeff_file=True,
                jacobian_file=True,
                modulatedref_file=True,
                out_intensitymap_file=True,
            ),
            name='fnirt',
        )
        preproc.connect(flirt, 'out_file', fnirt, 'in_file')

        identity = pe.Node(
            interface=IdentityInterface(
                fields=['struct_brain_warped'],
            ),
            name='identity',
        )
        preproc.connect(fnirt, 'warped_file', identity, 'struct_brain_warped')

        ### convert_fov = pe.Node(
        ###     interface=fsl.ConvertXFM(
        ###         invert_xfm=True,
        ###     ),
        ###     name='convert_fov',
        ### )
        ### preproc.connect(robust_fov, 'out_transform', convert_fov, 'in_file')

        convert_xfm = pe.Node(
            interface=fsl.ConvertXFM(
                invert_xfm=True,
            ),
            name='convert_xfm',
        )
        preproc.connect(flirt, 'out_matrix_file', convert_xfm, 'in_file')

        inv_warp = pe.Node(
            interface=fsl.InvWarp(),
            name='inv_warp',
        )
        preproc.connect(fnirt, 'field_file', inv_warp, 'warp')
        preproc.connect(flirt, 'out_file', inv_warp, 'reference')

        apply_warp = pe.Node(
            interface=fsl.ApplyWarp(
                in_file=mni_file,
            ),
            name='apply_warp'
        )
        preproc.connect(flirt, 'out_file', apply_warp, 'ref_file')
        preproc.connect(inv_warp, 'inverse_warp', apply_warp, 'field_file')

        apply_xfm = pe.Node(
            interface=fsl.ApplyXFM(
                apply_xfm=True,
            ),
            name='apply_xfm',
        )
        preproc.connect(fast, 'restored_image', apply_xfm, 'reference')
        preproc.connect(apply_warp, 'out_file', apply_xfm, 'in_file')
        preproc.connect(convert_xfm, 'out_file', apply_xfm, 'in_matrix_file')

        apply_fov = pe.Node(
            interface=fsl.ApplyXFM(
                apply_xfm=True,
            ),
            name='apply_fov',
        )
        preproc.connect(robust_fov, 'out_roi', apply_fov, 'reference')
        preproc.connect(apply_xfm, 'out_file', apply_fov, 'in_file')
        preproc.connect(robust_fov, 'out_transform', apply_fov, 'in_matrix_file')
        
        for i, atlas_file in enumerate([cerebellum_file, hocort_file, hosub_file]):
            warp_node = pe.Node(
                interface=fsl.ApplyWarp(in_file=atlas_file), 
                name='-'.join(atlas_file.name.split('-')[0:2]) + '_warp',
            )
            xfm_node = pe.Node(
                interface=fsl.ApplyXFM(in_file=atlas_file),
                name='-'.join(atlas_file.name.split('-')[0:2]) + '_xfm',
            )
            fov_node = pe.Node(
                interface=fsl.ApplyXFM(
                    in_file=atlas_file,
                    reference=img_file,
                ),
                name='-'.join(atlas_file.name.split('-')[0:2]) + '_fov',
            )
            preproc.connect(flirt, 'out_file', warp_node, 'ref_file')
            preproc.connect(inv_warp, 'inverse_warp', warp_node, 'field_file')
            preproc.connect(fast, 'restored_image', xfm_node, 'reference')
            preproc.connect(warp_node, 'out_file', xfm_node, 'in_file')
            preproc.connect(convert_xfm, 'out_file', xfm_node, 'in_matrix_file')
            preproc.connect(xfm_node, 'out_file', fov_node, 'in_file')
            preproc.connect(robust_fov, 'out_transform', fov_node, 'in_matrix_file')
        
        output = preproc.run()

        return {
            ### 'fast': onix_path / study_date / workfolder / './fast/struct_brain_seg.nii.gz',
            'fast': onix_path / './fast_sinker/brainmask.auto_out_seg.nii.gz',
            'cerebellum_inv': onix_path / workfolder / './Cerebellum-MNIfnirt_xfm/Cerebellum-MNIfnirt-maxprob-thr0-1mm_warp_flirt.nii.gz',
            'cortical_inv': onix_path / workfolder / './HarvardOxford-cort_xfm/HarvardOxford-cort-maxprob-thr0-1mm_warp_flirt.nii.gz',
            'subcortical_inv': onix_path / workfolder / './HarvardOxford-sub_xfm/HarvardOxford-sub-maxprob-thr0-1mm_warp_flirt.nii.gz',
        }

    def get_cho_naa_masks(self, fitt_threshold=2.0, nnfit_threshold=2.0, use_avg=False):
        fitt_cho_naa_mask, avg = self.nawm_threshold(
            self.fitt_cho_naa.image, threshold=fitt_threshold,
        )
        nnfit_cho_naa_mask, _ = self.nawm_threshold(
            self.nnfit_cho_naa.image, threshold=nnfit_threshold, avg=avg if use_avg else None,
        )
        return dict(
            fitt_cho_naa_mask = fitt_cho_naa_mask,
            nnfit_cho_naa_mask = nnfit_cho_naa_mask,
            #fitt_cho_naa_mask_1connect = fitt_cho_naa_mask.connected_component(),
            #nnfit_cho_naa_mask_1connect = nnfit_cho_naa_mask.connected_component(),
        )

    def get_cho_naa_metrics(
            self, 
            threshold=2.0, 
            overwrite_ccb=True, 
            threshold_fitt=None, 
            threshold_nnfit=None,
    ):
        """ """
        ###fitt_cho_naa = self.fitt_ds.load_sinorm_map("CHO/NAA")
        ###fitt_cho_naa = fitt_cho_naa.align(self.fitt_ds.t1)

        ###nnfit_cho_naa = self.nnfit_ds.load_nnfit_map("cho/naa ratio")
        ###nnfit_cho_naa = nnfit_cho_naa.register(self.fitt_ds.ref, self.ref_tx)
        ###nnfit_cho_naa = nnfit_cho_naa.align(self.fitt_ds.t1)

        ###self.fitt_cho_naa = fitt_cho_naa
        ###self.nnfit_cho_naa = nnfit_cho_naa

        if threshold_fitt is not None and threshold_nnfit is not None:
            self.fitt_cho_naa_mask, _ = self.nawm_threshold(
                self.fitt_cho_naa.image, threshold=threshold_fitt
            )
            self.nnfit_cho_naa_mask,_ = self.nawm_threshold(
                self.nnfit_cho_naa.image, threshold=threshold_nnfit
            )
        else:
            self.fitt_cho_naa_mask, _ = self.nawm_threshold(
                self.fitt_cho_naa.image, threshold=threshold
            )
            self.nnfit_cho_naa_mask,_ = self.nawm_threshold(
                self.nnfit_cho_naa.image, threshold=threshold
            )

        # TODO >>>
        self.fitt_cho_naa_mask_1connect = self.fitt_cho_naa_mask.connected_component()
        self.nnfit_cho_naa_mask_1connect = self.nnfit_cho_naa_mask.connected_component()

        #self.fitt_cho_naa_mask_2connect = copy.deepcopy(self.fitt_cho_naa_mask_1connect)
        #self.nnfit_cho_naa_mask_2connect = copy.deepcopy(self.nnfit_cho_naa_mask_1connect)

        # Don't take primary connected component before hand...
        self.fitt_cho_naa_mask_2connect = copy.deepcopy(self.fitt_cho_naa_mask)
        self.nnfit_cho_naa_mask_2connect = copy.deepcopy(self.nnfit_cho_naa_mask)
        # TODO <<<

        ###
        ### qmap filter
        ###

        def mask_filter(image, thresh=0.0):
            below_filter = itk.ThresholdImageFilter[itk.Image[itk.F, 3]].New()
            below_filter.SetInput(image)
            below_filter.ThresholdBelow(thresh)
            below_filter.SetOutsideValue(0.0)
            below_filter.Update()
            above_filter = itk.ThresholdImageFilter[itk.Image[itk.F, 3]].New()
            above_filter.SetInput(below_filter.GetOutput())
            above_filter.ThresholdAbove(0.0)
            above_filter.SetOutsideValue(1.0)
            above_filter.Update()
            return above_filter.GetOutput()

        qmap = self.fitt_ds.qmap.align(self.fitt_ds.t1)
        qmap_array = np.where(qmap.array == 4, 1, 0)
        qmap_image = mask_filter(qmap.image, thresh=3.5)
        qmap_image = cast_uc(qmap_image)

        def mask_map(image, mask_image):
            multiply_filter = itk.MultiplyImageFilter[
                image,
                mask_image,
                image,
            ].New()
            multiply_filter.SetCoordinateTolerance(1e-3)
            multiply_filter.SetDirectionTolerance(1e-3)
            try:
                multiply_filter.SetInput1(image)
                multiply_filter.SetInput2(mask_image)
                multiply_filter.Update()
                image = multiply_filter.GetOutput()
            except Exception as e:
                print(e)
                multiply_filter.SetInput1(image)
                multiply_filter.SetInput2(align_image(mask_image, image))
                multiply_filter.Update()
                image = multiply_filter.GetOutput()

            return image

        fitt_cho_naa_qmap = mask_map(
            self.fitt_cho_naa_mask_2connect.image, qmap_image,
        )
        nnfit_cho_naa_qmap = mask_map(
            self.nnfit_cho_naa_mask_2connect.image, qmap_image,
        )

        ###
        ### preprocess t1
        ###
        onix_path = self.fitt_ds.study.subject_path / 'onix'
        subject_id = self.fitt_ds.study.subject_id
        study_date = self.fitt_ds.study.date.replace('/', '_')
        _study_date = self.fitt_ds.study.date.replace('/', '-')
        ### workfolder = f"preprocess-t1_{subject_id}_{_study_date}"
        workfolder = "preprocess_t1"
        if (onix_path / workfolder).exists() and not overwrite_ccb:
            files = {
                ### 'fast': onix_path / study_date / workfolder / './fast/struct_brain_seg.nii.gz',
                'fast': onix_path / study_date / './fast_sinker/brainmask.auto_out_seg.nii.gz',
                'cerebellum_inv': onix_path / workfolder / './Cerebellum-MNIfnirt_xfm/Cerebellum-MNIfnirt-maxprob-thr0-1mm_warp_flirt.nii.gz',
                'cortical_inv': onix_path / workfolder / './HarvardOxford-cort_xfm/HarvardOxford-cort-maxprob-thr0-1mm_warp_flirt.nii.gz',
                'subcortical_inv': onix_path / workfolder / './HarvardOxford-sub_xfm/HarvardOxford-sub-maxprob-thr0-1mm_warp_flirt.nii.gz',
            }
        elif (onix_path / study_date / workfolder).exists() and not overwrite_ccb:
            files = {
                ### 'fast': onix_path / study_date / workfolder / './fast/struct_brain_seg.nii.gz',
                'fast': onix_path / study_date / './fast_sinker/brainmask.auto_out_seg.nii.gz',
                'cerebellum_inv': onix_path / study_date / workfolder / './Cerebellum-MNIfnirt_xfm/Cerebellum-MNIfnirt-maxprob-thr0-1mm_warp_flirt.nii.gz',
                'cortical_inv': onix_path / study_date / workfolder / './HarvardOxford-cort_xfm/HarvardOxford-cort-maxprob-thr0-1mm_warp_flirt.nii.gz',
                'subcortical_inv': onix_path / study_date / workfolder / './HarvardOxford-sub_xfm/HarvardOxford-sub-maxprob-thr0-1mm_warp_flirt.nii.gz',
            }
        else:
            #print('\n', onix_path / workfolder, (onix_path / workfolder).exists(), '\n')
            #print('\n', onix_path / study_date / workfolder, (onix_path / study_date / workfolder).exists(), '\n')
            #print('\n', 'overwrite_ccb', overwrite_ccb, '\n')
            files = self.preprocess_t1()
        self.cerebellum = cerebellum = itk.imread(files['cerebellum_inv'])
        self.brainstem = brainstem = itk.imread(files['subcortical_inv'])
        self.tissue = tissue = itk.imread(files['fast'])

        ###
        ### remove csf, cerebellum, and brainstem
        ###
        def ccb_filter(image, image2, image3):
            """"""
            caster = itk.CastImageFilter[image, itk.Image[itk.UC, 3]].New()
            caster.SetInput(image)
            caster.Update()
            image = caster.GetOutput()
            
            below_filter = itk.ThresholdImageFilter[image].New()
            below_filter.SetInput(image)
            below_filter.ThresholdAbove(0)
            below_filter.SetOutsideValue(1)
            below_filter.Update()
            image = below_filter.GetOutput()
        
            struc = itk.FlatStructuringElement[3].Ball(1)
            closing_filter = itk.BinaryMorphologicalClosingImageFilter[image, image, struc].New()
            closing_filter.SetInput(image)
            closing_filter.SetKernel(struc)
            closing_filter.SetForegroundValue(1)
            closing_filter.Update()
            image = closing_filter.GetOutput()
            
            caster = itk.CastImageFilter[image2, itk.Image[itk.UC, 3]].New()
            caster.SetInput(image2)
            caster.Update()
            image2 = caster.GetOutput()
            
            below_filter = itk.ThresholdImageFilter[image2].New()
            below_filter.SetInput(image2)
            below_filter.ThresholdOutside(8, 8)
            below_filter.SetOutsideValue(0)
            below_filter.Update()
            image2 = below_filter.GetOutput()
        
            struc = itk.FlatStructuringElement[3].Ball(1)
            opening_filter = itk.BinaryMorphologicalOpeningImageFilter[image2, image2, struc].New()
            opening_filter.SetInput(image2)
            opening_filter.SetKernel(struc)
            opening_filter.SetForegroundValue(8)
            opening_filter.Update()
            image2 = opening_filter.GetOutput()
            
            add_filter = itk.AddImageFilter[image, image2, image].New()
            add_filter.SetInput1(image)
            add_filter.SetInput2(image2)
            add_filter.Update()
            add_image = add_filter.GetOutput()
            
            above_filter = itk.ThresholdImageFilter[add_image].New()
            above_filter.SetInput(add_image)
            above_filter.ThresholdAbove(0)
            above_filter.SetOutsideValue(1)
            above_filter.Update()
            add_image = above_filter.GetOutput()
            
            struc = itk.FlatStructuringElement[3].Ball(3)
            closing_filter = itk.BinaryMorphologicalClosingImageFilter[image, image, struc].New()
            closing_filter.SetInput(add_image)
            closing_filter.SetKernel(struc)
            closing_filter.SetForegroundValue(1)
            closing_filter.Update()
            add_image = closing_filter.GetOutput()
            
            caster = itk.CastImageFilter[image3, itk.Image[itk.UC, 3]].New()
            caster.SetInput(image3)
            caster.Update()
            image3 = caster.GetOutput()
            
            below_filter = itk.ThresholdImageFilter[image3].New()
            below_filter.SetInput(image3)
            below_filter.ThresholdOutside(1, 1)
            below_filter.SetOutsideValue(0)
            below_filter.Update()
            image3 = below_filter.GetOutput()
            
            add_filter = itk.AddImageFilter[add_image, image3, add_image].New()
            add_filter.SetInput1(add_image)
            add_filter.SetInput2(image3)
            add_filter.Update()
            add_image = add_filter.GetOutput()
            
            above_filter = itk.ThresholdImageFilter[add_image].New()
            above_filter.SetInput(add_image)
            above_filter.ThresholdAbove(0)
            above_filter.SetOutsideValue(255)
            above_filter.Update()
            add_image = above_filter.GetOutput()
            
            below_filter = itk.ThresholdImageFilter[add_image].New()
            below_filter.SetInput(add_image)
            below_filter.ThresholdBelow(1)
            below_filter.SetOutsideValue(1)
            below_filter.Update()
            add_image = below_filter.GetOutput()
            
            above_filter = itk.ThresholdImageFilter[add_image].New()
            above_filter.SetInput(add_image)
            above_filter.ThresholdAbove(254)
            above_filter.SetOutsideValue(0)
            above_filter.Update()
            add_image = above_filter.GetOutput()
            
            return add_image

        ccb_mask = ccb_filter(cerebellum, brainstem, tissue)
        self.ccb_mask = cast_float(ccb_mask)
        self.ccb_mask = OnixVolume(
            itk.GetArrayFromImage(align_map(self.ccb_mask, self.fitt_ds.ref.image)),
            self.ccb_mask,
        )

        fitt_cho_naa_qmap_ccb = mask_map(
            fitt_cho_naa_qmap, ccb_mask,
        )
        nnfit_cho_naa_qmap_ccb = mask_map(
            nnfit_cho_naa_qmap, ccb_mask,
        )

        ###
        ### update volume
        ###

        self.fitt_cho_naa_mask_2connect.image = fitt_cho_naa_qmap_ccb
        self.nnfit_cho_naa_mask_2connect.image = nnfit_cho_naa_qmap_ccb

        self.fitt_cho_naa_mask_2connect.array = itk.GetArrayFromImage(fitt_cho_naa_qmap_ccb)
        self.nnfit_cho_naa_mask_2connect.array = itk.GetArrayFromImage(nnfit_cho_naa_qmap_ccb)

        ### 
        ### get primary connected component
        ###
        self.fitt_cho_naa_mask_2connect = self.fitt_cho_naa_mask_2connect.connected_component()
        self.nnfit_cho_naa_mask_2connect = self.nnfit_cho_naa_mask_2connect.connected_component()

        cho_naa_metrics = sg.write_metrics(
            labels=[
                1,
            ],
            #gdth_img=itk_to_sitk(self.fitt_cho_naa_mask.image),
            #pred_img=itk_to_sitk(self.nnfit_cho_naa_mask.image),
            ###gdth_img=itk_to_sitk(fitt_cho_naa_qmap),
            ###pred_img=itk_to_sitk(nnfit_chonaa_qmap),
            ###gdth_img=itk_to_sitk(self.fitt_cho_naa_mask_1connect.image),
            ###pred_img=itk_to_sitk(self.nnfit_cho_naa_mask_1connect.image),
            gdth_img=itk_to_sitk(self.fitt_cho_naa_mask_2connect.image),
            pred_img=itk_to_sitk(self.nnfit_cho_naa_mask_2connect.image),
        )

        return (
            {k: v[0] for k, v in cho_naa_metrics[0].items()} 
            | {
                'vfitt': np.sum(self.fitt_cho_naa_mask_2connect.array)/1000,
                'vnnfit': np.sum(self.nnfit_cho_naa_mask_2connect.array)/1000,
            }
        )

    def update_cho_naa_metrics(
        self, 
        fitt_threshold=2.0, 
        nnfit_threshold=2.0, 
    ):
        """ """
        return (
            pd.DataFrame.from_dict(
                {
                    "cho/naa update": self.get_cho_naa_metrics(
                        threshold_fitt=fitt_threshold,
                        threshold_nnfit=nnfit_threshold,
                        overwrite_ccb=False,
                    ),
                }
            )
            .T.round(4)
            .reset_index()
            .rename(columns={"index": "mask"})
        )

    def get_cho_naa_df(self, threshold=2.0, overwrite_ccb=True):
        """ """
        return (
            pd.DataFrame.from_dict(
                {
                    #'cho/naa 1.5x': self.get_cho_naa_metrics(threshold=1.5),
                    "cho/naa 2.0x": self.get_cho_naa_metrics(
                        threshold=2.0, 
                        overwrite_ccb=overwrite_ccb,
                    ),
                    #'cho/naa 2.5x': self.get_cho_naa_metrics(threshold=2.5),
                }
            )
            .T.round(4)
            .reset_index()
            .rename(columns={"index": "mask"})
        )

    def get_bounding_box(self, x):
        """Calculates the bounding box of a ndarray"""
        x = itk.GetArrayFromImage(x.image)
        mask = x == 0
        bbox = []
        all_axis = np.arange(x.ndim)
        for kdim in all_axis:
            nk_dim = np.delete(all_axis, kdim)
            mask_i = mask.all(axis=tuple(nk_dim))
            dmask_i = np.diff(mask_i)
            idx_i = np.nonzero(dmask_i)[0]
            if len(idx_i) != 2:
                raise ValueError(
                    "Algorithm failed, {} does not have 2 elements!".format(idx_i)
                )
            bbox.append(slice(idx_i[0] + 1, idx_i[1] + 1))
        return bbox

    def get_map_df(self):
        """ """
        metabolites = {
            "cho": "CHO",
            "cr": "CR",
            "naa": "NAA",
            "glx": "GLX",
            "mi": "MINO",
        }

        ratios = {
            "cho/naa ratio": "CHO/NAA",
        }

        ###if bound_box:
        ###    bbox = self.get_bounding_box(self.fitt_ds.brain_mask)

        output = {}
        output_t1 = {}

        for key, value in metabolites.items():
            try:
                vol1 = self.fitt_ds.load_si_map(f"{value}_Area")
                vol2 = self.nnfit_ds.load_nnfit_map(f"{key} area")
                vol2 = vol2.register(self.fitt_ds.ref, self.ref_tx)

                output[key] = self.compare_map_3d(
                    itk.GetArrayFromImage(vol1.image),
                    itk.GetArrayFromImage(vol2.image),
                )
                output[key] |= self.compare_map_correlation(
                    itk.GetArrayFromImage(vol1.image),
                    itk.GetArrayFromImage(vol2.image),
                )

                vol1 = vol1.align(self.fitt_ds.t1)
                vol2 = vol2.align(self.fitt_ds.t1)
                output_t1[key] = self.compare_map_3d_itk(
                    itk.GetArrayFromImage(vol1.image),
                    itk.GetArrayFromImage(vol2.image),
                )

            except Exception as e:
                print(f"WARNING: missing info for {key} vs {value} calculations")

            # TODO >>>
            try:
                if self.nnfit_ds._og:
                    _key = 'og_cho_naa' if 'ratio' in key else f'og_{key}'
                    vol3 = self.nnfit_ds.nnfit_ds.load_og_map(_key)
                    vol3 = vol3.register(self.fitt_ds.ref, self.ref_tx).align(self.fitt_ds.t1)
                    output_t1[_key] = self.compare_map_3d_itk(
                        itk.GetArrayFromImage(vol1.image),
                        itk.GetArrayFromImage(vol3.image),
                    )
                else:
                    print(f"\nWARNING: missing info for OG {key} vs {value} calculations\n")
            except Exception as e:
                print(f"\nWARNING: error for OG {key} vs {value} calculations\n{e}\n")
            # TODO <<<

        for key, value in ratios.items():
            try:
                vol1 = self.fitt_ds.load_sinorm_map(value)
                vol2 = self.nnfit_ds.load_nnfit_map(key)
                vol2 = vol2.register(self.fitt_ds.ref, self.ref_tx)

                output[key] = self.compare_map_3d(
                    itk.GetArrayFromImage(vol1.image),
                    itk.GetArrayFromImage(vol2.image),
                )
                output[key] |= self.compare_map_correlation(
                    itk.GetArrayFromImage(vol1.image),
                    itk.GetArrayFromImage(vol2.image),
                )

                vol1 = vol1.align(self.fitt_ds.t1)
                vol2 = vol2.align(self.fitt_ds.t1)
                output_t1[key] = self.compare_map_3d_itk(
                    itk.GetArrayFromImage(vol1.image),
                    itk.GetArrayFromImage(vol2.image),
                )
            except Exception as e:
                print(f"WARNING: missing info for {key} vs {value} calculations")

            # TODO >>>
            try:
                if self.nnfit_ds._og:
                    _key = 'og_cho_naa' if 'ratio' in key else f'og_{key}'
                    vol3 = self.nnfit_ds.nnfit_ds.load_og_map(_key)
                    vol3 = vol3.register(self.fitt_ds.ref, self.ref_tx).align(self.fitt_ds.t1)
                    output_t1[_key] = self.compare_map_3d_itk(
                        itk.GetArrayFromImage(vol1.image),
                        itk.GetArrayFromImage(vol3.image),
                    )
                else:
                    print(f"\nWARNING: missing info for OG {key} vs {value} calculations\n")
            except Exception as e:
                print(f"\nWARNING: error for OG {key} vs {value} calculations\n{e}\n")
            # TODO <<<

        df = (
            pd.DataFrame.from_dict(output)
            .T.round(4)
            .reset_index()
            .rename(columns={"index": "metabolite"})
        )
        df_t1 = (
            pd.DataFrame.from_dict(output_t1)
            .T.round(4)
            .reset_index()
            .rename(columns={"index": "metabolite"})
        )

        return df, df_t1

    def get_cho_naa_table(self, threshold=2.0, overwrite_ccb=True):
        """ """
        return self.get_table(self.get_cho_naa_df(threshold=threshold, overwrite_ccb=overwrite_ccb))

    def get_map_table(self):
        """ """
        return self.get_table(self.get_map_df())
