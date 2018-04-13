#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDisplacementFieldTransform.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageSeriesWriter.h"
#include "itkNumericSeriesFileNames.h"
#include "itkCastImageFilter.h"
#include "itkTIFFImageIO.h"
#include "itkNrrdImageIO.h"
#include "itkImageIOBase.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkPermuteAxesImageFilter.h"
#include "itkFlipImageFilter.h"
#include "itkAffineTransform.h"



int main(int argc, char *argv[])
{
  if( argc < 4 )
    {
      std::cerr << "Usage: " << std::endl;
      std::cerr << argv[0] << " inimage defField outimage" << std::endl; //CHANGED
      return EXIT_FAILURE;
    }
  // The input deformation field needs to be a displacement field converted from index coordinates
  //Define the dimension of the images that will be used and the pixel type (any)
  const unsigned int Dimension = 3;
  typedef float PixelType;
  typedef unsigned short OutPixelType;
  typedef float VectorComponentType;
  // const char * outputFileName=argv[3]; 
  // std::string format= std::string( outputFileName ) + std::string( "%d.tif" );

  //Define the different types of images
  typedef itk::Vector< VectorComponentType, Dimension > VectorPixelType;
  typedef itk::Image< VectorPixelType, Dimension > DeformationFieldType;
  typedef itk::Image< PixelType, Dimension > ImageType;
  typedef itk::Image< OutPixelType, 3 > UnsignedCharImageType;
  // typedef itk::Image< OutPixelType,2 > OutImageType;
  
  //Set up the reader and writer for the images
  typedef itk::ImageFileReader< ImageType > ImageReaderType;
  typedef itk::ImageFileWriter< OutImageType > ImageWriterType;
  typedef itk::ImageFileReader< DeformationFieldType > FieldReaderType;
  typedef itk::TIFFImageIO TIFFIOType;
  //typedef itk::ImageSeriesWriter< ImageType,OutImageType > SeriesWriterType;

  //Not really sure what this line acutally does. But it is used. 
  itk::NrrdImageIO::Pointer nrrd = itk::NrrdImageIO::New();


  ImageReaderType::Pointer reader = ImageReaderType::New();
  FieldReaderType::Pointer fieldReader = FieldReaderType::New();
  ImageWriterType::Pointer tiffwriter = ImageWriterType::New();
  ImageWriterType::Pointer nrrdWriter = ImageWriterType::New();
  TIFFIOType::Pointer tiffIO = TIFFIOType::New();

  reader->SetFileName( argv[1] );
  fieldReader->SetFileName( argv[2] );
  nrrdWriter->SetFileName( argv[3] + std::string( ".nrrd" ));
  tiffwriter->SetFileName( argv[3] + std::string( ".tif" ));


  //Read in the images and fields
  std::cout << "Attempting to read the input image volume...";
  try
    {
      reader->Update();
    }
  catch(itk::ExceptionObject & err)
    {
      std::cerr << "ExceptionObject caught !"<< std::endl;
      std::cerr << err <<std::endl;
      return EXIT_FAILURE;
    }
  std::cout << "Success" << std::endl;

  //ImageType::Pointer inputImage = reader->GetOutput();

  //Read in the displacement field
  std::cout << "Attempting to read the input deformation volume...";
  try
    {
      fieldReader->Update();
    }
  catch(itk::ExceptionObject & err)
    {
      std::cerr << "ExceptionObject caught !"<< std::endl;
      std::cerr << err <<std::endl;
      return EXIT_FAILURE;
    }

  std::cout << "Success" << std::endl << std::endl;

  DeformationFieldType::Pointer deformationField = fieldReader->GetOutput();

  // Set up interpolator
  typedef itk::LinearInterpolateImageFunction< ImageType,float > InterpolatorType;
  InterpolatorType::Pointer interpolator=InterpolatorType::New();
  typedef itk::ResampleImageFilter< ImageType,ImageType,float > ResampleFilterType;
  

  //Flip the image along the first axis and rotate by 90 degrees (this changes from TIFF to MHA)
  typedef itk::FlipImageFilter< ImageType > FlipperType;
  FlipperType::Pointer flipper=FlipperType::New();
  flipper->SetInput(reader->GetOutput());

  FlipperType::FlipAxesArrayType flipAxes;
  flipAxes[0] = false;
  flipAxes[1] = false;

  flipper->SetFlipAxes(flipAxes);
  flipper->Update();

  ImageType::Pointer temp = flipper->GetOutput();

  // Set the origin of the image back to 0.
  ImageType::PointType flipOrg;
  flipOrg[0] = 0.0;
  flipOrg[1] = 0.0;
  flipOrg[2] = 0.0;

  ImageType::SpacingType flipSpc;
  flipSpc[0] = 1.0;
  flipSpc[1] = 1.0;
  flipSpc[2] = 1.0;

  temp->SetOrigin( flipOrg );
  temp->SetSpacing( flipSpc );
  temp->Update();

  // // ResampleFilterType::PointType test;
  // // flipOrg[0] = -2349.0/2.0;
  // // flipOrg[1] = 2349.0/2.0;
  // // flipOrg[2] = 0.0;
  
  typedef itk::AffineTransform< float, 3 >  TransformType;
  TransformType::Pointer transform = TransformType::New();

  //Set the size of the image following rotation
  ResampleFilterType::SizeType rotSize;
  rotSize[0] = 53566.0;//52703.0; //5859.0;
  rotSize[1] = 76345.0;//73866.0; //8208.0;
  rotSize[2] = 1.0;

  TransformType::InputPointType center;
  center[0] = 76345.0/2.0;//73866.0/2.0; //8209.0/2.0;
  center[1] = 53566.0/2.0;//52703.0/2.0; //5859.0/2.0;
  center[2] = 0;

  
  ResampleFilterType::Pointer rotate = ResampleFilterType::New();
  rotate->SetInterpolator( interpolator );
  rotate->SetDefaultPixelValue( 0.0 );

  const double degreesToRadians = atan(1.0) / 45.0;
  const double angle = 90.0 * degreesToRadians;

  TransformType::InputVectorType axis;
  axis[0]=0.0;
  axis[1]=0.0;
  axis[2]=1.0;

  transform->SetCenter( center );
  transform->Rotate3D( axis,angle,false );

  // Has to do with rotating a rectang around its center
  const double test[ Dimension ] = { 9576.5, -9576.5, 0.0 };

  
  rotate->SetTransform( transform );
  rotate->SetOutputOrigin( test );
  rotate->SetSize(rotSize);
  rotate->SetInput( temp );
  rotate->Update();

  
  ImageType::Pointer inputImage = temp;


  // Set the spacing and the origin of the input image accordingly. 
  ImageType::SizeType sizeSetterInput;
  sizeSetterInput[0] = 76345.0;//52703.0; //5859.0;
  sizeSetterInput[1] = 53566.0;//73866.0; //8208.0;
  sizeSetterInput[2] = 1.0;

  ImageType::SpacingType spacingSetterInput;
  spacingSetterInput[0] = (2048.0/sizeSetterInput[0])*deformationField->GetSpacing()[0];
  spacingSetterInput[1] = (2048.0/sizeSetterInput[1])*deformationField->GetSpacing()[1];
  // spacingSetterOutput[2] = (1.0/sizeSetterOutput[2])*deformationField->GetSpacing()[2];
  spacingSetterInput[2] = 1.0;

  ImageType::PointType originSetterInput;
  originSetterInput[0] = -(sizeSetterInput[0]*spacingSetterInput[0])/2; 
  originSetterInput[1] = -(sizeSetterInput[1]*spacingSetterInput[1])/2;
  originSetterInput[2] = 0.0; //-(sizeSetterOutput[2]*spacingSetterOutput[2])/2;



  // Set the size of the desired image in MRI space
  ImageType::SizeType sizeSetterOutput;
  sizeSetterOutput[0] = 53566.0;//52703.0; //5859.0;
  sizeSetterOutput[1] = 76345.0;//73866.0; //8208.0;
  sizeSetterOutput[2] = 1.0;

  // Find the spacing that is required in MRI space 
  ImageType::SpacingType spacingSetterOutput;
  spacingSetterOutput[0] = (2048.0/sizeSetterOutput[0])*deformationField->GetSpacing()[0];
  spacingSetterOutput[1] = (2048.0/sizeSetterOutput[1])*deformationField->GetSpacing()[1];
  // spacingSetterOutput[2] = (1.0/sizeSetterOutput[2])*deformationField->GetSpacing()[2];
  spacingSetterOutput[2] = 1.0;


  ImageType::PointType originSetter;
  originSetter[0] = -(sizeSetterOutput[0]*spacingSetterOutput[0])/2; 
  originSetter[1] = -(sizeSetterOutput[1]*spacingSetterOutput[1])/2;
  originSetter[2] = 0.0; //-(sizeSetterOutput[2]*spacingSetterOutput[2])/2;
  
  inputImage->SetSpacing( spacingSetterInput );
  inputImage->SetOrigin( originSetterInput );
  inputImage->Update();

  // std::cout << "Input Image Origin = " << std::endl;
  // std::cout << tempImage->GetDirection() << std::endl;
  std::cout << "Input Image Origin = " << inputImage->GetOrigin() << std::endl;
  std::cout << "Input Image Spacing = " << inputImage->GetSpacing() << std::endl;
  std::cout << "Input Image size = " << inputImage->GetBufferedRegion().GetSize() << std::endl << std::endl;
  
  std::cout << "Deformation Field Origin = " << deformationField->GetOrigin() << std::endl;
  std::cout << "Deformation Field Spacing = " << deformationField->GetSpacing() << std::endl;
  std::cout << "Deformation Field Size = " << deformationField->GetBufferedRegion().GetSize() << std::endl << std::endl;
  
  typedef itk::DisplacementFieldTransform< VectorComponentType,Dimension > DeformationFieldTransformType;
  DeformationFieldTransformType::Pointer deformationFieldTransform = DeformationFieldTransformType::New();
  deformationFieldTransform->SetDisplacementField( deformationField );
  

  //Define the resample filter
  ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

    // Set the parameters of the resample filter
  resampleFilter->SetSize( sizeSetterOutput );
  resampleFilter->SetOutputOrigin( originSetter );
  //  resampleFilter->SetOutputOrigin( deformationField->GetOrigin() );
  resampleFilter->SetOutputSpacing( spacingSetterOutput );

  // Set final parameters of the resample filter and then update
  resampleFilter->SetInterpolator( interpolator );
  resampleFilter->SetDefaultPixelValue( 0.0 );
  resampleFilter->SetTransform( deformationFieldTransform );
  resampleFilter->SetInput( inputImage );
  resampleFilter->SetNumberOfThreads( 100 );
  resampleFilter->Update();

  typedef itk::RescaleIntensityImageFilter< ImageType, OutImageType > RescalerType;
  RescalerType::Pointer rescaler=RescalerType::New();

  rescaler->SetOutputMinimum( 0 );
  rescaler->SetOutputMaximum( 65535 );
  //rescaler->SetInput(resampleFilter->GetOutput());  
  rescaler->SetInput(resampleFilter->GetOutput());
  
  OutImageType::Pointer finalImage=rescaler->GetOutput();


  std::cout << "Final Image Origin = " << resampleFilter->GetOutputOrigin() << std::endl;
  std::cout << "Final Image Spacing = " << resampleFilter->GetOutputSpacing() << std::endl;
  std::cout << "Final Image Size = " << resampleFilter->GetSize() << std::endl << std::endl;


  //Set the inputs to the writers
  tiffwriter->SetImageIO(tiffIO);
  tiffwriter->SetInput(finalImage);  

  nrrdWriter->SetImageIO( nrrd );
  nrrdWriter->SetInput( finalImage );

  
  std::cout << "Attempting to write:" << std::endl;  
  
  try
    {

      tiffwriter->Update();

    }
  catch(itk::ExceptionObject & err)
    {
      std::cerr << "ExceptionObject caught !"<< std::endl;
      std::cerr << err <<std::endl;
      return EXIT_FAILURE;
    }
  std::cout << "Successfully Wrote out TIFF Image" << std::endl;
  
  try
    {
      nrrdWriter->Update();

    }
  catch(itk::ExceptionObject & err)
    {
      std::cerr << "ExceptionObject caught !"<< std::endl;
      std::cerr << err <<std::endl;
      return EXIT_FAILURE;
    }


  std::cout << "Successfully Wrote out NRRD Image" << std::endl;  
  return EXIT_SUCCESS;
}

  // //Stuff for writing image series
  // ImageType::SizeType size=finalImage->GetLargestPossibleRegion().GetSize();
  // ImageType::IndexType start=finalImage->GetLargestPossibleRegion().GetIndex();

  // itk::NumericSeriesFileNames::Pointer fnames = itk::NumericSeriesFileNames::New(); \\Changed
  // fnames->SetSeriesFormat( format.c_str() );
  // fnames->SetStartIndex( start[2] );
  // fnames->SetEndIndex( start[2] + size[2]-1);
  // fnames->SetIncrementIndex( 1 );
