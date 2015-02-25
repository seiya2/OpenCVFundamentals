#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

namespace cv {

	class FaceDetection : public CascadeClassifier
	{

	public:
		void detectMultiScaleMultiCamera( const Mat& image, vector<Rect>& objects,
											  vector<int>& rejectLevels,
											  vector<double>& levelWeights,
											  double scaleFactor=1.1, int minNeighbors=3,
											  int flags=0, Size minObjectSize=Size(), Size maxObjectSize=Size(),
											  bool outputRejectLevels=false );
		void detectMultiScaleMultiCamera( const Mat& image, vector<Rect>& objects,
											  double scaleFactor=1.1, int minNeighbors=3,
											  int flags=0, Size minObjectSize=Size(), Size maxObjectSize=Size());
	};
}

