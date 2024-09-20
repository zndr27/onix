import dash
from dash import dcc, html, dash_table, Dash
###from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc

import copy
import glob
import itk
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import plotly.express as px
from scipy.signal import resample
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import seg_metrics.seg_metrics as sg
import skimage
from skimage.metrics import (
    mean_squared_error,
    normalized_root_mse,
    structural_similarity,
)
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
)
import time
from typing_extensions import Self
import xarray as xr

from nnfit.xtra.dash_slicer import VolumeSlicer
from nnfit.data.midas import *
from nnfit.utils.image import *
from nnfit.utils.image import _register_routine


class OnixViewer(OnixObject):
    """ """

    _PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
    _DBC_CSS = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css"
    _EXTERNAL_STYLESHEETS = [dbc.themes.DARKLY, _DBC_CSS]

    subject_id: str
    study_date: str

    ###app: Dash | JupyterDash
    app: Dash

    slicer_z: VolumeSlicer
    slicer_y: VolumeSlicer
    slicer_x: VolumeSlicer

    fitt_ds: OnixDataset
    nnfit_ds: OnixDataset
    ds: dict[str, OnixDataset]

    t1_tx = None
    ref_tx = None

    metrics: OnixMetrics

    overlay_list: list[str]
    overlay: OnixOverlay

    def __init__(
        self,
        fitt_subject_xml: Path,
        nnfit_subject_xml: Path,
        extra_subject_xml: list[Path],
        study_date: str,
        jupyter: bool = False,
        flip_x=False,
        log=False,
        overwrite_outputs=False,
        overwrite_ccb=True,
        save_metrics_path=None,
        og=False,
    ):
        # Load outputs
        sub_path = nnfit_subject_xml.parent
        onix_path = sub_path / 'onix' / study_date.replace('/','_')

        save_outputs = True

        if onix_path.is_dir() and not overwrite_outputs:
            # Load transformations
            files = list(x.name for x in onix_path.iterdir())
            if (
                'fitt_tx.pkl' in files and
                'nnfit_tx.pkl' in files and
                't1_tx.pkl' in files and
                'ref_tx.pkl' in files and
                "map_df.pkl" in files and
                "map_df_t1.pkl" in files and
                "map_table.pkl" in files and
                "map_table_t1.pkl" in files and
                "ratio_df.pkl" in files and
                "ratio_table.pkl" in files and 
                "ccb_mask.pkl" in files
            ):
                save_outputs = False
                print('\nloading registrations\n')

        if save_outputs:
            print('\ngenerating registrations\n')

        # TODO update with force overwrite
        os.makedirs(onix_path, exist_ok=True)
        
        _profile_time = time.time()

        self.subject_id = fitt_subject_xml.parent.name
        self.study_date = study_date

        fitt_study = MidasSubject(fitt_subject_xml).study(study_date)
        nnfit_study = MidasSubject(nnfit_subject_xml).study(study_date)

        print("time 0:", _profile_time - time.time())
        _profile_time = time.time()

        self.fitt_ds = OnixDataset(
            fitt_study, 
            "fitt", 
            log=log, 
            save_outputs=save_outputs,
            og=og,
        )
        _fitt_ds_time = time.time()
        self.nnfit_ds = OnixDataset(
            nnfit_study, 
            "nnfit", 
            flip_x=flip_x, 
            log=log, 
            save_outputs=save_outputs,
            og=og,
        )
        _nnfit_ds_time = time.time()

        print("time 1:", _profile_time - time.time())
        print("\tfitt_ds:", _profile_time - _fitt_ds_time)
        print("\tnnfit_ds:", _fitt_ds_time - _nnfit_ds_time)
        _profile_time = time.time()

        self.overlay_list = (
            ["None", "Reference", "Brain Mask"]
            + list(self.fitt_ds.si_maps.keys())
            + list(self.fitt_ds.sinorm_maps.keys())
            + list(self.fitt_ds.seg_masks.keys())
            + self.nnfit_ds.nnfit_maps
            + ([] if not og else self.nnfit_ds.og_maps)
        )

        if save_outputs:

            print("REGISTER ROUTINES")

            _, self.t1_tx = _register_routine(
                self.fitt_ds.t1.image,
                self.nnfit_ds.t1.image,
                learn_rate=0.01,
                stop=0.001,
                max_steps=50,
                log=log,
            )

            with open(onix_path / 't1_tx.pkl', 'wb') as f:
                pickle.dump(self.t1_tx, f, pickle.HIGHEST_PROTOCOL)

            _, self.ref_tx = _register_routine(
                self.fitt_ds.ref.image,
                self.nnfit_ds.ref.image,
                learn_rate=0.01,
                stop=0.001,
                max_steps=50,
                log=log,
            )

            with open(onix_path / 'ref_tx.pkl', 'wb') as f:
                pickle.dump(self.ref_tx, f, pickle.HIGHEST_PROTOCOL)

        else:

            print("LOADING REGISTRATION ROUTINES") 

            with open(onix_path / 't1_tx.pkl', 'rb') as f:
                self.t1_tx = pickle.load(f)
            with open(onix_path / 'ref_tx.pkl', 'rb') as f:
                self.ref_tx = pickle.load(f)

        print("time 2:", _profile_time - time.time())
        _profile_time = time.time()

        ###self.nnfit_ds.spectra  = self.nnfit_ds.spectra.register(self.fitt_ds.ref, self.nnfit_ds.ref, self.ref_tx)
        ###self.nnfit_ds.baseline = self.nnfit_ds.baseline.register(self.fitt_ds.ref, self.nnfit_ds.ref, self.ref_tx)
        ###self.nnfit_ds.fit      = self.nnfit_ds.fit.register(self.fitt_ds.ref, self.nnfit_ds.ref, self.ref_tx)

        print("time 3:", _profile_time - time.time())
        _profile_time = time.time()

        _metrics_time = time.time()
        self.metrics = OnixMetrics(
            self.fitt_ds,
            self.nnfit_ds,
            self.t1_tx,
            self.ref_tx,
            self.fitt_ds.brain_mask,
            self.fitt_ds.qmap,
            save_outputs = save_outputs,
            overwrite_ccb = overwrite_ccb,
        )
        print("metrics time:", _metrics_time - time.time())

        # Initialize cho/naa masks
        self.update_cho_naa_threshold(2.0, 2.0)

        _overlay_time = time.time()
        self.overlay = OnixOverlay(
            ref=self.fitt_ds.ref,
            brain_mask=self.fitt_ds.brain_mask.align(self.fitt_ds.t1),
            qmap=self.fitt_ds.qmap.align(self.fitt_ds.t1),
            hqmap=self.fitt_ds.hqmap.align(self.fitt_ds.t1),
            vhqmap=self.fitt_ds.vhqmap.align(self.fitt_ds.t1),
            nnqmap=self.fitt_ds.nnqmap.align(self.fitt_ds.t1),
            nnhqmap=self.fitt_ds.nnhqmap.align(self.fitt_ds.t1),
            nnvhqmap=self.fitt_ds.nnvhqmap.align(self.fitt_ds.t1),
            t2star=self.fitt_ds.t2star.align(self.fitt_ds.t1),
            ccb_mask=self.metrics.ccb_mask,
            volume_1=None,
            volume_2=None,
        )
        # From metrics...
        self.overlay_list += ['fitt cho/naa mask', 'nnfit cho/naa mask']
        #self.overlay_list += ['fitt cho/naa mask connect', 'nnfit cho/naa mask connect']
        self.overlay_list += ['fitt cho/naa 2x', 'nnfit cho/naa 2x']
        self.overlay_list += ['fitt cho/naa 2x connect', 'nnfit cho/naa 2x connect']
        self.overlay_list += ['fitt cho/naa 2x connect2', 'nnfit cho/naa 2x connect2']
        print("overlay time:", _overlay_time - time.time())

        print("time 4:", _profile_time - time.time())
        _profile_time = time.time()

        ###if jupyter:
        ###    self.app = JupyterDash(
        ###        __name__,
        ###        external_stylesheets=self._EXTERNAL_STYLESHEETS,
        ###        use_pages=True,
        ###        # suppress_callback_exceptions=True,
        ###    )
        ###else:
        ###    self.app = Dash(
        ###        __name__,
        ###        external_stylesheets=self._EXTERNAL_STYLESHEETS,
        ###        use_pages=True,
        ###        # suppress_callback_exceptions=True,
        ###    )
        self.app = Dash(
            __name__,
            external_stylesheets=self._EXTERNAL_STYLESHEETS,
            use_pages=True,
            pages_folder="",
            # suppress_callback_exceptions=True,
        )

        print("time 5:", _profile_time - time.time())
        _profile_time = time.time()

        self.init_slicer()
        self.init_pages()
        self.init_callbacks()

        print("time 6:", _profile_time - time.time())
        _profile_time = time.time()

        # SAVE METRICS
        if save_metrics_path is not None:
            os.makedirs(save_metrics_path, exist_ok=True)
            self.metrics.save_metrics(save_metrics_path)

    def update_overlay_volume(self, label, second=False):
        """ """
        self.overlay.threshold = True

        if label == "None":
            if second:
                self.overlay.update_volume_2(None)
            else:
                self.overlay.update_volume_1(None)
            return

        elif label == "Reference":
            self.overlay.threshold = False
            volume = self.fitt_ds.ref

        elif label == "Brain Mask":
            self.overlay.threshold = False
            volume = self.fitt_ds.brain_mask

        elif label in self.nnfit_ds.nnfit_maps:
            volume = self.nnfit_ds.load_nnfit_map(label)
            volume = volume.register(self.fitt_ds.ref, self.ref_tx)

        elif label in self.fitt_ds.si_maps.keys():
            volume = self.fitt_ds.load_si_map(label)

        elif label in self.fitt_ds.sinorm_maps.keys():
            volume = self.fitt_ds.load_sinorm_map(label)

        elif label in self.fitt_ds.seg_masks.keys():
            volume = self.fitt_ds.load_seg_mask(label)

        elif self.nnfit_ds._og and label in self.nnfit_ds.og_maps:
            volume = self.nnfit_ds.nnfit_ds.load_og_map(label)
        
        elif label == "fitt cho/naa mask":
            self.overlay.threshold = False
            volume = self.fitt_cho_naa_mask

        elif label == "nnfit cho/naa mask":
            self.overlay.threshold = False
            volume = self.nnfit_cho_naa_mask

        elif label == "fitt cho/naa 2x":
            self.overlay.threshold = False
            volume = self.metrics.fitt_cho_naa_mask

        elif label == "nnfit cho/naa 2x":
            self.overlay.threshold = False
            volume = self.metrics.nnfit_cho_naa_mask
        
        elif label == "fitt cho/naa 2x connect":
            self.overlay.threshold = False
            volume = self.metrics.fitt_cho_naa_mask_1connect

        elif label == "nnfit cho/naa 2x connect":
            self.overlay.threshold = False
            volume = self.metrics.nnfit_cho_naa_mask_1connect
        
        elif label == "fitt cho/naa 2x connect2":
            self.overlay.threshold = False
            volume = self.metrics.fitt_cho_naa_mask_2connect

        elif label == "nnfit cho/naa 2x connect2":
            self.overlay.threshold = False
            volume = self.metrics.nnfit_cho_naa_mask_2connect

        else:
            raise Exception("Invalid overlay volume")

        # Align to the t1 image.
        volume = volume.align(self.fitt_ds.t1)

        if second:
            self.overlay.update_volume_2(volume)
        else:
            self.overlay.update_volume_1(volume)

    def update_overlay_scaling(self, scaling):
        """ """
        self.overlay.update_scaling(scaling)

    def update_overlay_operation(self, operation):
        """ """
        self.overlay.update_operation(operation)

    def update_overlay_colormap(self, color_range: tuple, alpha: float, fixed_range: tuple):
        """ """
        self.overlay.update_colormap(color_range, alpha, fixed_range)

    def update_overlay_mask(self, mask: str):
        """ """
        self.overlay.update_mask(mask)

    def update_cho_naa_threshold(self, fitt_threshold: float, nnfit_threshold: float, use_avg=False):
        """ """
        masks = self.metrics.get_cho_naa_masks(
            fitt_threshold = fitt_threshold,
            nnfit_threshold = nnfit_threshold,
            use_avg = use_avg,
        )
        self.fitt_cho_naa_mask = masks.get('fitt_cho_naa_mask')
        self.nnfit_cho_naa_mask = masks.get('nnfit_cho_naa_mask')
        #self.fitt_cho_naa_mask_1connect = masks.get('fitt_cho_naa_mask_1connect')
        #self.nnfit_cho_naa_mask_1connect = masks.get('nnfit_cho_naa_mask_1connect')

    def update_overlay(self):
        """ """
        _overlay = self.overlay.get_slicer(update=True)
        _colormap = self.overlay.get_colormap()
        return (
            self.slicer_z.create_overlay_data(_overlay, _colormap),
            self.slicer_y.create_overlay_data(_overlay, _colormap),
            self.slicer_x.create_overlay_data(_overlay, _colormap),
        )

    def si_to_mri_coords(self, x: float, y: float, z: float):
        """ """
        return self.fitt_ds.t1.image.TransformPhysicalPointToIndex(
            self.fitt_ds.ref.image.TransformIndexToPhysicalPoint([x, y, z])
        )

    def mri_to_si_coords(self, x: float, y: float, z: float):
        """ """
        return self.fitt_ds.ref.image.TransformPhysicalPointToIndex(
            self.fitt_ds.t1.image.TransformIndexToPhysicalPoint([x, y, z])
        )

    def fitt_to_nnfit_coords(self, x, y, z):
        """ """
        return self.nnfit_ds.ref.image.TransformPhysicalPointToIndex(
            list(
                self.ref_tx.TransformPoint(
                    list(
                        self.fitt_ds.ref.image.TransformIndexToPhysicalPoint([x, y, z])
                    )
                )
            )
        )

    def init_slicer(self):
        """ """
        array = self.fitt_ds.t1.slicer_array()

        self.slicer_z = VolumeSlicer(
            self.app,
            array,
            axis=0,
            thumbnail=False,
        )
        self.slicer_y = VolumeSlicer(
            self.app,
            array,
            axis=1,
            thumbnail=False,
        )
        self.slicer_x = VolumeSlicer(
            self.app,
            array,
            axis=2,
            thumbnail=False,
        )

        self.slicer_z.graph.config["scrollZoom"] = False
        self.slicer_y.graph.config["scrollZoom"] = False
        self.slicer_x.graph.config["scrollZoom"] = False

        self.slicer_z.graph.figure.update_layout(template="plotly_dark")
        self.slicer_y.graph.figure.update_layout(template="plotly_dark")
        self.slicer_x.graph.figure.update_layout(template="plotly_dark")

    def init_pages(self):
        """ """
        from nnfit.onix.layouts import main_layout, metrics_layout, slicer_layout

        dash.register_page("metrics", path="/metrics", layout=metrics_layout(self))
        dash.register_page("slicer", path="/", layout=slicer_layout(self))
        self.app.layout = main_layout(self)

    def init_callbacks(self):
        """ """
        from nnfit.onix.callbacks import metrics_callbacks, slicer_callbacks

        metrics_callbacks(self)
        slicer_callbacks(self)

    def start(
        self,
        debug=True,
        dev_tools_props_check=False,
        host="0.0.0.0",
        port=8050,
        **kwargs,
    ):
        self.app.run_server(
            debug=debug,
            dev_tools_props_check=dev_tools_props_check,
            host=host,
            port=port,
            **kwargs,
        )
