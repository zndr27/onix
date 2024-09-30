import lxml.etree as ET
import numpy as np
from pathlib import Path
from pymidas.common.libxml import ProjectXml, SubjectXml
import SimpleITK as sitk
import zlib

from onix.utils.data import *


class MidasNode:
    """ """

    id: None | str
    node: ET._Element
    subject_xml: SubjectXml
    subject_id: str

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        self.id = None
        self.node = node
        self.subject_xml = subject_xml
        self.subject_path = subject_path
        self.subject_id = subject_xml.get_subject_id()

    def all_param(self):
        """ """
        # TODO: loop is redundant
        params = {}
        param_list = [
            (x.get("name"), x.get("value")) for x in self.node.xpath(f"./param")
        ]
        for p in param_list:
            name, value = p[0], p[1]
            params[name] = [x[1] for x in param_list if x[0] == name]
            if len(params[name]) == 1:
                params[name] = params[name][0]
        return params

    def param(self, name):
        """ """
        return self.subject_xml.get_parameter_given_id(self.id, name)


class MidasFrame(MidasNode):
    """ """

    path: Path

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Frame_ID']/@value")[0]
        self.path = self.subject_path / self.subject_xml.get_file_path_given_id(self.id).replace('\\', '/')

    def load(self):
        """ """
        offset = int(self.param("Byte_Offset"))

        data_type = self.param("Data_Representation")
        dtype, ImageInputType = parse_data_type(data_type)

        px = float(self.param("Image_Position_X"))
        py = float(self.param("Image_Position_Y"))
        pz = float(self.param("Image_Position_Z"))
        dx = int(self.param("Spatial_Points_1"))
        dy = int(self.param("Spatial_Points_2"))
        dz = int(self.param("Spatial_Points_3"))
        sx = float(self.param("Pixel_Spacing_1"))
        sy = float(self.param("Pixel_Spacing_2"))
        sz = float(self.param("Pixel_Spacing_3"))
        oxr = float(self.param("Image_Orientation_Xr"))
        oyr = float(self.param("Image_Orientation_Yr"))
        ozr = float(self.param("Image_Orientation_Zr"))
        oxc = float(self.param("Image_Orientation_Xc"))
        oyc = float(self.param("Image_Orientation_Yc"))
        ozc = float(self.param("Image_Orientation_Zc"))

        array = np.fromfile(self.path, dtype=dtype, count=dx * dy * dz, offset=offset)
        array = array.reshape(dz, dy, dx)
        image = sitk.GetImageFromArray(array)

        image.SetSpacing([sx, sy, sz])
        image.SetOrigin(update_origin(px, py, pz, oxr, oyr, ozr, oxc, oyc, ozc))
        image.SetDirection(
            np.array(
                [
                    [
                        oxr,
                        oyr,
                        ozr,
                    ],
                    [
                        oxc,
                        oyc,
                        ozc,
                    ],
                    cross_product(oxr, oyr, ozr, oxc, oyc, ozc),
                ],
                dtype=np.float64,
            ).flatten()
        )

        if data_type != "float":
            image = cast_float(image)

        return array, image


class MidasInput(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Input_ID']/@value")[0]
        self.output_data_id = self.param("Output_Data_ID")

    def data(self):
        """ """
        return MidasData(
            self.node.xpath(
                f"../data[./param[@name='Data_ID' and @value='{self.output_data_id}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )


class MidasData(MidasNode):
    """ """

    path: Path

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Data_ID']/@value")[0]
        self.path = self.subject_path / self.subject_xml.get_file_path_given_id(self.id).replace('\\', '/')

    def all_frame(self):
        """ """
        return [
            MidasFrame(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./frame")
        ]

    def frame(self, frame_type):
        """ """
        return MidasFrame(
            self.node.xpath(
                f"./frame[./param[@name='Frame_Type' and @value='{frame_type}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )

    def load_all_frames(self):
        """ """
        frame_type_list = self.node.xpath(f"./frame/param[@name='Frame_Type']/@value")
        frames = {frame_type: self.load_frame(frame_type)}
        return frames

    def load_frame(self, frame_type=None):
        """ """
        if frame_type == None:
            frame = MidasFrame(
                self.node.xpath(f"./frame")[0], self.subject_xml, self.subject_path
            )
        else:
            frame = MidasFrame(
                self.node.xpath(
                    f"./frame[param[@name='Frame_Type' and @value='{frame_type}']"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        return frame.load()

    def load_image(self):
        """ """
        data_type = self.param("Data_Representation")
        dtype, ImageInputType = parse_data_type(data_type)

        px = float(self.param("Image_Position_X"))
        py = float(self.param("Image_Position_Y"))
        pz = float(self.param("Image_Position_Z"))
        dx = int(self.param("Spatial_Points_1"))
        dy = int(self.param("Spatial_Points_2"))
        dz = int(self.param("Spatial_Points_3"))
        sx = float(self.param("Pixel_Spacing_1"))
        sy = float(self.param("Pixel_Spacing_2"))
        sz = float(self.param("Pixel_Spacing_3"))
        oxr = float(self.param("Image_Orientation_Xr"))
        oyr = float(self.param("Image_Orientation_Yr"))
        ozr = float(self.param("Image_Orientation_Zr"))
        oxc = float(self.param("Image_Orientation_Xc"))
        oyc = float(self.param("Image_Orientation_Yc"))
        ozc = float(self.param("Image_Orientation_Zc"))

        array = np.fromfile(self.path, dtype=dtype, count=dx * dy * dz)
        array = array.reshape(dz, dy, dx)
        image = sitk.GetImageFromArray(array)

        image.SetSpacing([sx, sy, sz])
        image.SetOrigin(update_origin(px, py, pz, oxr, oyr, ozr, oxc, oyc, ozc))
        image.SetDirection(
            np.array(
                [
                    [
                        oxr,
                        oyr,
                        ozr,
                    ],
                    [
                        oxc,
                        oyc,
                        ozc,
                    ],
                    cross_product(oxr, oyr, ozr, oxc, oyc, ozc),
                ],
                dtype=np.float64,
            ).flatten()
        )

        if data_type != "float":
            image = cast_float(image)

        return array, image

    def load_spectra(self):
        """ """
        if self.param("Compression") and self.param("Compression").lower() == "zlib":
            with open(self.path, "rb") as f:
                spec = np.frombuffer(zlib.decompress(f.read()), dtype=np.float32)
        else:
            spec = np.fromfile(self.path, dtype=np.float32)

        spec = spec.reshape(
            int(self.param("Spatial_Points_3")),
            int(self.param("Spatial_Points_2")),
            int(self.param("Spatial_Points_1")),
            int(self.param("Spectral_Points_1")),
            2,
        )

        return spec[..., 0] + 1j * spec[..., 1]


class MidasProcess(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Process_ID']/@value")[0]

    def dataset(self, created_by):
        """ """
        if created_by == None:
            return MidasDataset(
                self.node.xpath(f"./dataset")[0], self.subject_xml, self.subject_path
            )
        else:
            return MidasDataset(
                self.node.xpath(
                    f"./dataset[./param[@name='Created_By' and @value='{created_by}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )

    def all_input(self):
        """ """
        return [
            MidasInput(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./input")
        ]

    def input(self, process_name):
        """ """
        return MidasInput(
            self.node.xpath(
                f"./input[./param[@name='Process_Name' and @value='{process_name}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )

    def all_data(self):
        """ """
        return [
            MidasData(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./data")
        ]

    def data(self, created_by=None, frame_type=None):
        """ """
        if created_by == None and frame_type == None:
            return MidasData(
                self.node.xpath(f"./data")[0], self.subject_xml, self.subject_path
            )
        elif created_by:
            return MidasData(
                self.node.xpath(
                    f"./data[./param[@name='Created_By' and @value='{created_by}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        elif frame_type:
            return MidasData(
                self.node.xpath(
                    f"./data[./frame/param[@name='Frame_Type' and @value='{frame_type}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )

    def all_frame(self):
        """ """
        return [
            MidasFrame(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./data/frame")
        ]

    def frame(self, frame_type):
        """ """
        return MidasFrame(
            self.node.xpath(
                f"./data/frame[./param[@name='Frame_Type' and @value='{frame_type}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )


class MidasDataset(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Dataset_ID']/@value")[0]

    def all_data(self):
        """ """
        return [
            MidasData(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./data")
        ]

    def data(self, created_by=None):
        """ """
        if created_by == None:
            return MidasData(
                self.node.xpath(f"./data")[0], self.subject_xml, self.subject_path
            )
        else:
            return MidasData(
                self.node.xpath(
                    f"./data[./param[@name='Created_By' and @value='{created_by}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )

    def all_frame(self):
        """ """
        return [
            MidasFrame(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./data/frame")
        ]

    def frame(self, frame_type):
        """ """
        return MidasFrame(
            self.node.xpath(
                f"./data/frame[./param[@name='Frame_Type' and @value='{frame_type}']]"
            )[0],
            self.subject_xml,
            self.subject_path,
        )


class MidasSeries(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Series_ID']/@value")[0]

    def all_process(self):
        """ """
        return [
            MidasProcess(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./process")
        ]

    def process(self, label):
        """ """
        try:
            return MidasProcess(
                self.node.xpath(f"./process[./param[@name='Label' and @value='{label}']]")[
                    0
                ],
                self.subject_xml,
                self.subject_path,
            )
        except IndexError:
            return None

    def all_dataset(self):
        """ """
        return [
            MidasDataset(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./dataset")
        ]

    def dataset(self, label):
        """ """
        return MidasDataset(
            self.node.xpath(f"./dataset[./param[@name='Label' and @value='{label}']]")[
                0
            ],
            self.subject_xml,
            self.subject_path,
        )


class MidasStudy(MidasNode):
    """ """

    def __init__(self, node: ET._Element, subject_xml: SubjectXml, subject_path: Path):
        super().__init__(node, subject_xml, subject_path)
        self.id = self.node.xpath(f"./param[@name='Study_ID']/@value")[0]
        self.date = self.param("Study_Date")

    def all_series(self):
        """ """
        return [
            MidasSeries(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./series")
        ]

    def series(self, label=None, series_id=None):
        """ """
        if label:
            return MidasSeries(
                self.node.xpath(
                    f"./series[./param[@name='Label' and @value='{label}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        elif series_id:
            return MidasSeries(
                self.node.xpath(
                    f"./series[./param[@name='Series_ID' and @value='{series_id}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        else:
            raise Exception(f"No label or series_id specified")

    def process(self, label):
        """ """
        return MidasProcess(
            self.node.xpath(f"./process[./param[@name='Label' and @value='{label}']]")[
                0
            ],
            self.subject_xml,
            self.subject_path,
        )

    def t1(self):
        """ """
        data = self.series("MRI_T1").process("Volume").data()
        return data.load_image()

    def flair(self):
        """ """
        data = self.series("MRI_FLAIR").process("Volume").data()
        return data.load_image()

    def ref(self):
        """ """
        data = self.series("SI_Ref").process("Maps").data()
        return data.load_image()

    def brain_mask(self):
        """ """
        frame = self.series("SI_Ref").process("Maps").frame("Mask_Brain")
        return frame.load()

    def lipid_mask(self):
        """ """
        frame = self.series("SI_Ref").process("Maps").frame("Mask_Lipid")
        return frame.load()

    def si(self):
        """ """
        return self.series("SI").process("Spectral").data().load_spectra()

    def siref(self):
        """ """
        return self.series("SI_Ref").process("Spectral").data().load_spectra()

    def spectral_sampling(self, node):
        """ """
        hz_per_ppm = float(node.param("Precession_Frequency"))
        spec_pts = int(node.param("Spectral_Points_1"))
        freq_offset = float(node.param("Frequency_Offset"))
        chem_shift_ref = float(node.param("Chemical_Shift_Reference"))
        spec_width = float(node.param("Spectral_Width_1"))
        hz_per_pt = spec_width / spec_pts
        ppm_range = spec_width / hz_per_ppm
        ppm_per_pt = ppm_range / spec_pts
        center_ppm = chem_shift_ref + freq_offset
        left_edge_ppm = center_ppm + (ppm_range / 2)

        return dict(
            hz_per_ppm=hz_per_ppm,
            spec_pts=spec_pts,
            freq_offset=freq_offset,
            chem_shift_ref=chem_shift_ref,
            spec_width=spec_width,
            hz_per_pt=hz_per_pt,
            ppm_range=ppm_range,
            ppm_per_pt=ppm_per_pt,
            center_ppm=center_ppm,
            left_edge_ppm=left_edge_ppm,
        )

    def si_sampling(self):
        """ """
        data = self.series("SI").process("Spectral").data()
        return self.spectral_sampling(data)

    def siref_sampling(self):
        """ """
        data = self.series("SI_Ref").process("Spectral").data()
        return self.spectral_sampling(data)

    def fitt(self):
        """ """
        if not self.series("SI").process("Spectral_FitBase"):
            return None
        return self.series("SI").process("Spectral_FitBase").data().load_spectra()

    def fitt_baseline(self):
        """ """
        if not self.series("SI").process("Spectral_BL"):
            return None
        return self.series("SI").process("Spectral_BL").data().load_spectra()

    def qmap(self):
        """ """
        frame = self.series("SI").process("Maps").data("QMaps").frame("Quality_Map")
        return frame.load()

    def t2star(self):
        """ """
        frame = self.series("SI_Ref").process("Maps").frame("T2Star_Map")
        return frame.load()

    def segmentation(self, label):
        """ """
        data = self.process("MRI_SEG").dataset("MriSeg").data(label)
        return data.load_frame()


class MidasSubject(MidasNode):
    """ """

    def __init__(self, subject_xml: Path):
        """ """
        subject_xml = Path(subject_xml)
        super().__init__(
            ET.parse(
                subject_xml, parser=ET.XMLParser(remove_blank_text=True)
            ).getroot(),
            SubjectXml(str(subject_xml)),
            subject_xml.parent,
        )
        self.id = self.node.xpath(f"./param[@name='Subject_ID']/@value")[0]

    def all_study(self):
        """ """
        return [
            MidasStudy(x, self.subject_xml, self.subject_path)
            for x in self.node.xpath(f"./study")
        ]

    def study(self, study_date, study_time=None):
        """ """
        if study_time == None:
            return MidasStudy(
                self.node.xpath(
                    f"./study[./param[@name='Study_Date' and @value='{study_date}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )
        else:
            return MidasStudy(
                self.node.xpath(
                    f"./study"
                    f"[./param[@name='Study_Date' and @value='{study_date}']]"
                    f"[./param[@name='Study_Time' and @value='{study_time}']]"
                )[0],
                self.subject_xml,
                self.subject_path,
            )


class MidasProject:
    """ """

    def __init__(self, project_xml: Path):
        """ """
        project_xml = Path(project_xml)
        self.node = ET.parse(
            project_xml, parser=ET.XMLParser(remove_blank_text=True)
        ).getroot()
        # self.project_xml = ProjectXml(str(project_xml))
        self.project_xml = project_xml
        self.path = project_xml.parent
        self.name = self.node.xpath(f"./param[@name='Project_Name']/@value")[0]

    def all_subject(self):
        """ """
        subject_nodes = self.node.xpath(f"./Subject")
        subject_list = []
        for node in subject_nodes:
            subject_dir = node.xpath(f"./param[@name='Subject_Directory']/@value")[0]
            subject_xml = self.path / subject_dir / "subject.xml"
            subject_list.append(MidasSubject(subject_xml))
        return subject_list

    def subject(self, subject_id):
        """ """
        return MidasSubject(self.path / subject_id / "subject.xml")


def midas_array_to_sitk(node: MidasData, array: np.ndarray) -> sitk.Image:
    """
    Convert a numpy array to an sitk image using MidasData object.
    """
    # Extract the metadata for the image
    px = float(self.data.param("Image_Position_X"))
    py = float(self.data.param("Image_Position_Y"))
    pz = float(self.data.param("Image_Position_Z"))
    dx = int(self.data.param("Spatial_Points_1"))
    dy = int(self.data.param("Spatial_Points_2"))
    dz = int(self.data.param("Spatial_Points_3"))
    sx = float(self.data.param("Pixel_Spacing_1"))
    sy = float(self.data.param("Pixel_Spacing_2"))
    sz = float(self.data.param("Pixel_Spacing_3"))
    oxr = float(self.data.param("Image_Orientation_Xr"))
    oyr = float(self.data.param("Image_Orientation_Yr"))
    ozr = float(self.data.param("Image_Orientation_Zr"))
    oxc = float(self.data.param("Image_Orientation_Xc"))
    oyc = float(self.data.param("Image_Orientation_Yc"))
    ozc = float(self.data.param("Image_Orientation_Zc"))

    # Pass the raw data to an itk image
    image = itk.GetImageFromArray(array)

    # Update image with the metadata
    image.SetSpacing([sx, sy, sz])
    image.SetOrigin(update_origin(px, py, pz, oxr, oyr, ozr, oxc, oyc, ozc))
    image.SetDirection(
        np.array(
            [
                [oxr, oyr, ozr],
                [oxc, oyc, ozc],
                cross_product(oxr, oyr, ozr, oxc, oyc, ozc),
            ],
            dtype=np.float64,
        )
    )

    # Cast image to float, since image operations are float
    if array.dtype != np.float32:
        image = cast_float(image)

    return image
