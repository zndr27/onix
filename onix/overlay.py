from Onix.objects import OnixObject


class OnixOverlay(OnixObject):
    """ """

    _alpha = 255
    _range = [0, 1000]
    _cmax = 1000
    _cmid = 500

    brain_mask: OnixVolume
    qmap: OnixVolume | None
    hqmap: OnixVolume | None
    vhqmap: OnixVolume | None
    nnqmap: OnixVolume | None
    t2star: OnixVolume | None
    volume_1: OnixVolume | None
    volume_2: OnixVolume | None

    _array: np.ndarray
    _image: itk.Image
    _slicer: np.ndarray

    mask_list = [
        "Brain Mask",
        "QMap",
        "hqmap",
        "vhqmap",
        "nnqmap",
        "nnhqmap",
        "nnvhqmap",
        "T2*",
        "ccb mask",
    ]
    slicer_mask = "Brain Mask"

    scaling_list = [
        "None",
        "Normal",
    ]
    normal: bool = False
    threshold: bool = True

    operation_list = [
        "None",
        "Add",
        "Divide",
        "Difference",
        "SSIM",
    ]
    ssim: bool = False
    difference: bool = False
    divide: bool = False
    add: bool = False

    def __init__(
        self,
        ref: OnixVolume,
        brain_mask: OnixVolume,
        qmap: OnixVolume = None,
        hqmap: OnixVolume = None,
        vhqmap: OnixVolume = None,
        nnqmap: OnixVolume = None,
        nnhqmap: OnixVolume = None,
        nnvhqmap: OnixVolume = None,
        t2star: OnixVolume = None,
        ccb_mask: OnixVolume = None,
        volume_1: OnixVolume = None,
        volume_2: OnixVolume = None,
    ):
        self.ref = ref
        self.brain_mask = brain_mask
        self.qmap = qmap
        self.hqmap = hqmap
        self.vhqmap = vhqmap
        self.nnqmap = nnqmap
        self.nnhqmap = nnhqmap
        self.nnvhqmap = nnvhqmap
        self.t2star = t2star
        self.ccb_mask = ccb_mask
        self.volume_1 = volume_1
        self.volume_2 = volume_2
        self._array = None
        self._image = None
        self._mask_array = None
        self._mask_image = None
        self._fixed_range = None

    def update_colormap(self, color_range, alpha, fixed_range):
        """ """
        if fixed_range is None:
            self._fixed_range = None
        else:
            self._fixed_range = fixed_range
        self._range = color_range
        self._alpha = alpha

    def update_mask(self, mask: str):
        """ """
        self.slicer_mask = mask

    def get_colormap(self):
        """ """
        if self.difference:
            cs = px.colors.get_colorscale("RdBu")
        else:
            #cs = px.colors.get_colorscale("Jet")
            cs = px.colors.get_colorscale("Turbo")
        cs = px.colors.sample_colorscale(cs, np.arange(0, 254) / 254)
        cs = [
            list(map(int, x.replace("rgb(", "").replace(")", "").split(", ")))
            + [
                self._alpha,
            ]
            for x in cs
        ]
        return cs

    def get_array(self, update):
        """ """
        if update:
            self.update()
        return self._array

    def get_image(self, update):
        """ """
        if update:
            self.update()
        return self._image

    def get_slicer(self, update):
        """ """
        if update:
            self.update()
        return self._slicer

    def get_value(self):
        """ """
        return self._array[z, y, x]

    def get_volume_1(self, x, y, z):
        """ """
        if self.volume_1 == None:
            return np.nan
        else:
            return self.volume_1.array[z, y, x]

    def get_volume_2(self, x, y, z):
        """ """
        if self.volume_2 == None:
            return np.nan
        else:
            return self.volume_2.array[z, y, x]

    def update_volume_1(self, volume: OnixVolume | None):
        """ """
        self.volume_1 = volume

    def update_volume_2(self, volume: OnixVolume | None):
        """ """
        self.volume_2 = volume

    def update_scaling(self, scaling="None"):
        """ """
        if scaling == "None":
            self.normal = False
        elif scaling == "Normal":
            self.normal = True
        else:
            raise Exception(f"Invalid scaling parameter")

    def update_operation(self, operation="None"):
        """ """
        self.ssim = False
        self.divide = False
        self.difference = False
        self.add = False
        if operation == "None":
            pass
        elif operation == "SSIM":
            self.ssim = True
        elif operation == "Difference":
            self.difference = True
        elif operation == "Divide":
            self.divide = True
        elif operation == "Add":
            self.add = True
        else:
            raise Exception(f"Invalid operation")

    def update(self):
        """ """
        if self.volume_1 == None:
            self._slicer = None
            return

        array1 = self.volume_1.array
        image1 = self.volume_1.image

        if self.volume_2 is not None:
            array2 = self.volume_2.array
            image2 = self.volume_2.image

        if self.volume_1.is_mask and self.volume_2 is not None and self.volume_2.is_mask:
            array1 *= 75
            array2 *= 150
            image1 = self.multiply_filter(image1, 75)
            image2 = self.multiply_filter(image2, 150)

        if self.slicer_mask == "T2*":
            # TODO: hz = 1/(np.pi*t2star)
            pass
        if self.slicer_mask == "QMap":
            mask_array = np.where(self.qmap.array == 4, 1, 0)
            mask_image = self.mask_filter(self.qmap.image, thresh=3.5)
        elif self.slicer_mask == "hqmap":
            mask_array = self.hqmap.array
            mask_image = self.mask_filter(self.hqmap.image, thresh=0.5)
        elif self.slicer_mask == "vhqmap":
            mask_array = self.vhqmap.array
            mask_image = self.mask_filter(self.vhqmap.image, thresh=0.5)
        elif self.slicer_mask == "nnqmap":
            mask_array = self.nnqmap.array
            mask_image = self.mask_filter(self.nnqmap.image, thresh=0.5)
        elif self.slicer_mask == "nnhqmap":
            mask_array = self.nnhqmap.array
            mask_image = self.mask_filter(self.nnhqmap.image, thresh=0.5)
        elif self.slicer_mask == "nnvhqmap":
            mask_array = self.nnvhqmap.array
            mask_image = self.mask_filter(self.nnvhqmap.image, thresh=0.5)
        elif self.slicer_mask == "ccb mask":
            mask_array = self.ccb_mask.array
            mask_image = self.mask_filter(self.ccb_mask.image, thresh=0.5)
        else:
            mask_array = self.brain_mask.array
            mask_image = self.mask_filter(self.brain_mask.image, thresh=0.5)

        if self.ssim and self.volume_2 != None:
            _, array = structural_similarity(
                self.volume_1.array,
                self.volume_2.array,
                data_range=1,
                full=True,
            )
            _, S = structural_similarity(
                itk.GetArrayFromImage(self.volume_1.image),
                itk.GetArrayFromImage(self.volume_2.image),
                win_size=21,
                data_range=1,
                full=True,
            )
            image = orient_array(S, self.volume_1.image)

        elif self.difference and self.volume_2 != None:
            array = array1 - array2
            image = self.difference_filter(image1, image2)

        elif self.divide and self.volume_2 != None:
            array = array1 / array2
            image = self.divide_filter(image1, image2)

        elif self.add and self.volume_2 != None:
            array = array1 + array2
            image = self.addition_filter(image1, image2)
        else:
            array = array1
            image = image1

        if self.normal:
            hist = array[np.where(self.brain_mask.array == 1)]
            mean = np.mean(hist)
            std = np.std(hist)
            array = (array - mean) / std
            image = self.normal_filter(image)

        # TODO: if the overlay is a mask then skip the colormap / histogram
        self._array = array
        self._image = image
        self._mask_array = mask_array
        self._mask_image = mask_image

        if self.volume_1.is_mask or (self.volume_2 is not None and self.volume_2.is_mask):
            self._xlo = np.min(self._array)
            self._xhi = np.max(self._array)
        else:
            self.rescale_map()
            
        self.mask_map()

        cast_filter = itk.CastImageFilter[type(self._image), itk.Image[itk.UC, 3]].New()
        cast_filter.SetInput(self._image)
        cast_filter.Update()

        self._slicer = slicer_array(cast_filter.GetOutput())

    def get_hist(self):
        """ """
        if self._slicer is None:
            return None, None, None
        else:
            hist = self._array[np.where(self._mask_array == 1)]
            return hist, self._xlo, self._xhi

    def rescale_map(self):
        """ """
        cmid = self._cmid
        cmax = self._cmax

        array = self._array
        image = self._image
        mask_array = self._mask_array
        mask_image = self._mask_image

        lo, hi = self._range
        if mask_array.shape != array.shape:
            raise Exception(
                f"Mask array shape {mask_array.shape} does not match image array shape {array.shape}"
            )
        hist = array[np.where(mask_array == 1)]
        xmin = np.min(hist)
        xmax = np.max(hist)

        rescale_filter = itk.RescaleIntensityImageFilter[
            itk.Image[itk.F, 3], itk.Image[itk.F, 3]
        ].New()

        if self.difference:
            xrm = max(-xmin, xmax)
            xhi = xrm * abs(hi - cmid) / cmid
            xlo = xrm * abs(cmid - lo) / cmid
            xr = min(xhi, xlo)
            array = np.where(array < xr, array, xr)
            array = np.where(array > -xr, array, -xr)
            image = self.threshold_above_filter(image, xr)
            image = self.threshold_below_filter(image, -xr)
            _min = np.min(array)
            _max = np.max(array)
            rescale_filter.SetInput(image)
            rescale_filter.SetOutputMinimum(128 + 127 * (_min / xr))
            rescale_filter.SetOutputMaximum(128 + 127 * (_max / xr))
            self._xlo = -xr
            self._xhi = xr
        else:
            if self._fixed_range is not None:
                xmin = self._fixed_range[0]
                xmax = self._fixed_range[1]
            #xmin = min(0, xmin)
            #xmax = max(16, xmax)
            xrange = xmax - xmin
            xhi = xmin + xrange * (hi / cmax)
            xlo = xmin + xrange * (lo / cmax)
            xr = xhi - xlo
            array = np.where(array < xhi, array, xhi)
            array = np.where(array > xlo, array, xlo)
            image = self.threshold_above_filter(image, xhi)
            image = self.threshold_below_filter(image, xlo)
            _min = np.min(array)
            _max = np.max(array)
            rescale_filter.SetInput(image)
            #rescale_filter.SetOutputMinimum(1)
            #rescale_filter.SetOutputMaximum(255)
            rescale_filter.SetOutputMinimum(1 + 254 * ((_min - xlo) / xr))
            rescale_filter.SetOutputMaximum(1 + 254 * ((_max - xlo) / xr))
            self._xlo = xlo
            self._xhi = xhi

        rescale_filter.Update()
        image = rescale_filter.GetOutput()

        self._array = array
        self._image = image

    def mask_map(self):
        """ """
        multiply_filter = itk.MultiplyImageFilter[
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            self._image,
            self._mask_image,
            self._image,
        ].New()
        try:
            multiply_filter.SetInput1(self._image)
            multiply_filter.SetInput2(self._mask_image)
            multiply_filter.Update()
        except Exception as e:
            multiply_filter.SetInput1(self._image)
            multiply_filter.SetInput2(align_map(self._mask_image, self._image))
            multiply_filter.Update()
        self._image = multiply_filter.GetOutput()

    def mask_filter(self, image, thresh=0.0):
        """ """
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
    
    def anti_mask_filter(self, image, thresh=0.0):
        """ """
        anti_filter = itk.ThresholdImageFilter[itk.Image[itk.F, 3]].New()
        anti_filter.SetInput(image)
        anti_filter.ThresholdAbove(thresh)
        anti_filter.SetOutsideValue(-1.0)
        anti_filter.Update()
        below_filter = itk.ThresholdImageFilter[itk.Image[itk.F, 3]].New()
        below_filter.SetInput(image)
        below_filter.ThresholdAbove(-1.0)
        below_filter.SetOutsideValue(1.0)
        below_filter.Update()
        above_filter = itk.ThresholdImageFilter[itk.Image[itk.F, 3]].New()
        above_filter.SetInput(below_filter.GetOutput())
        above_filter.ThresholdBelow(0.0)
        above_filter.SetOutsideValue(0.0)
        above_filter.Update()
        return above_filter.GetOutput()

    def threshold_above_filter(self, image, thresh):
        """ """
        threshold_filter = itk.ThresholdImageFilter[itk.Image[itk.F, 3]].New()
        threshold_filter.SetInput(image)
        threshold_filter.ThresholdAbove(thresh)
        threshold_filter.SetOutsideValue(thresh)
        threshold_filter.Update()
        return threshold_filter.GetOutput()

    def threshold_below_filter(self, image, thresh):
        """ """
        threshold_filter = itk.ThresholdImageFilter[itk.Image[itk.F, 3]].New()
        threshold_filter.SetInput(image)
        threshold_filter.ThresholdBelow(thresh)
        threshold_filter.SetOutsideValue(thresh)
        threshold_filter.Update()
        return threshold_filter.GetOutput()

    def multiply_filter(self, image_1, image_2):
        """ """
        mul_filter = itk.MultiplyImageFilter[
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            image_1,
            image_1,
            image_1,
        ].New()
        mul_filter.SetInput1(image_1)
        mul_filter.SetInput2(image_2)
        mul_filter.Update()
        return mul_filter.GetOutput()

    def divide_filter(self, image_1, image_2):
        div_filter = itk.DivideImageFilter[
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            image_1,
            image_1,
            image_1,
        ].New()
        div_filter.SetInput1(image_1)
        div_filter.SetInput2(image_2)
        div_filter.Update()
        return div_filter.GetOutput()

    def addition_filter(self, image_1, image_2):
        """ """
        add_filter = itk.AddImageFilter[
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            image_1,
            image_1,
            image_1,
        ].New()
        add_filter.SetInput1(image_1)
        add_filter.SetInput2(image_2)
        add_filter.Update()
        return add_filter.GetOutput()

    def difference_filter(self, image_1, image_2):
        """ """
        subtract_filter = itk.SubtractImageFilter[
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            image_1,
            image_1,
            image_1,
        ].New()
        subtract_filter.SetInput1(image_1)
        subtract_filter.SetInput2(image_2)
        subtract_filter.Update()
        return subtract_filter.GetOutput()

    def normal_filter(self, image):
        """ """
        normal_filter = itk.NormalizeImageFilter[
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            image,
            image,
        ].New()
        normal_filter.SetInput(image)
        normal_filter.Update()
        return normal_filter.GetOutput()

    def exp_filter(self, image):
        """ """
        exp_filter = itk.ExpImageFilter[
            #itk.Image[itk.F, 3],
            #itk.Image[itk.F, 3],
            image,
            image,
        ].New()
        exp_filter.SetInput(image)
        exp_filter.Update()
        return exp_filter.GetOutput()
