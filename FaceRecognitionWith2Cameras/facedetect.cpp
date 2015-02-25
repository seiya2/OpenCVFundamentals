#include "facedetect.h"



namespace cv {
	struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };

	void FaceDetection::detectMultiScaleMultiCamera( const Mat& image, vector<Rect>& objects,
											  vector<int>& rejectLevels,
											  vector<double>& levelWeights,
											  double scaleFactor, int minNeighbors,
											  int flags, Size minObjectSize, Size maxObjectSize,
											  bool outputRejectLevels )
	{
		const double GROUP_EPS = 0.2;

		CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

		if( empty() )
			return;

		if( isOldFormatCascade() )
		{
			MemStorage storage(cvCreateMemStorage(0));
			CvMat _image = image;
			CvSeq* _objects = cvHaarDetectObjectsForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
												  minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
			vector<CvAvgComp> vecAvgComp;
			Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
			objects.resize(vecAvgComp.size());
			std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
			return;
		}

		objects.clear();

		if (!maskGenerator.empty()) {
			maskGenerator->initializeMask(image);
		}


		if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
			maxObjectSize = image.size();

		Mat grayImage = image;
		if( grayImage.channels() > 1 )
		{
			Mat temp;
			cvtColor(grayImage, temp, CV_BGR2GRAY);
			grayImage = temp;
		}

		Mat imageBuffer(image.rows + 1, image.cols + 1, CV_8U);
		vector<Rect> candidates;

		for( double factor = 1; ; factor *= scaleFactor )
		{
			Size originalWindowSize = getOriginalWindowSize();

			Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
			Size scaledImageSize( cvRound( grayImage.cols/factor ), cvRound( grayImage.rows/factor ) );
			Size processingRectSize( scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height );

			if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )
				break;
			if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height )
				break;
			if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
				continue;

			Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
			resize( grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );

			int yStep;
			if( getFeatureType() == cv::FeatureEvaluator::HOG )
			{
				yStep = 4;
			}
			else
			{
				yStep = factor > 2. ? 1 : 2;
			}

			int stripCount, stripSize;

			const int PTS_PER_THREAD = 1000;
			stripCount = ((processingRectSize.width/yStep)*(processingRectSize.height + yStep-1)/yStep + PTS_PER_THREAD/2)/PTS_PER_THREAD;
			stripCount = std::min(std::max(stripCount, 1), 100);
			stripSize = (((processingRectSize.height + stripCount - 1)/stripCount + yStep-1)/yStep)*yStep;

			if( !detectSingleScale( scaledImage, stripCount, processingRectSize, stripSize, yStep, factor, candidates,
				rejectLevels, levelWeights, outputRejectLevels ) )
				break;
		}


		objects.resize(candidates.size());
		std::copy(candidates.begin(), candidates.end(), objects.begin());

		if( outputRejectLevels )
		{
			groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
		}
		else
		{
			groupRectangles( objects, minNeighbors, GROUP_EPS );
		}
	}

	void FaceDetection::detectMultiScaleMultiCamera( const Mat& image, vector<Rect>& objects,
											  double scaleFactor, int minNeighbors,
											  int flags, Size minObjectSize, Size maxObjectSize)
	{
		vector<int> fakeLevels;
		vector<double> fakeWeights;
		detectMultiScale( image, objects, fakeLevels, fakeWeights, scaleFactor,
			minNeighbors, flags, minObjectSize, maxObjectSize, false );
	}
}