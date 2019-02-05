import SimpleITK as sitk


def N4BiasFieldCorrection(image, destination, nb_iteration=50):
    inputImage = sitk.ReadImage(image)

    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(nb_iteration)

    output = corrector.Execute(inputImage)
    sitk.WriteImage(output, destination)
