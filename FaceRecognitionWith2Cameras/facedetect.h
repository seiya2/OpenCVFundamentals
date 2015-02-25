#include <opencv2/core/core.hpp>
#include <opencv2/core/internal.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
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

	class HaarDetectObjects_ScaleImage_Invoker : public ParallelLoopBody
	{
	public:
		HaarDetectObjects_ScaleImage_Invoker( const CvHaarClassifierCascade* _cascade,
											  int _stripSize, double _factor,
											  const Mat& _sum1, const Mat& _sqsum1, Mat* _norm1,
											  Mat* _mask1, Rect _equRect, std::vector<Rect>& _vec,
											  std::vector<int>& _levels, std::vector<double>& _weights,
											  bool _outputLevels, Mutex *_mtx );

		void operator()( const Range& range ) const;

		const CvHaarClassifierCascade* cascade;
		int stripSize;
		double factor;
		Mat sum1, sqsum1, *norm1, *mask1;
		Rect equRect;
		std::vector<Rect>* vec;
		std::vector<int>* rejectLevels;
		std::vector<double>* levelWeights;
		Mutex* mtx;
	};

	class HaarDetectObjects_ScaleCascade_Invoker : public ParallelLoopBody
	{
	public:
		HaarDetectObjects_ScaleCascade_Invoker( const CvHaarClassifierCascade* _cascade,
												Size _winsize, const Range& _xrange, double _ystep,
												size_t _sumstep, const int** _p, const int** _pq,
												std::vector<Rect>& _vec, Mutex* _mtx );

		void operator()( const Range& range ) const;

		const CvHaarClassifierCascade* cascade;
		double ystep;
		size_t sumstep;
		Size winsize;
		Range xrange;
		const int** p;
		const int** pq;
		std::vector<Rect>* vec;
		Mutex* mtx;
	};

	template<typename T, typename ST, typename QT>
	void integralMultiImage_( const T* src, size_t _srcstep, ST* sum, size_t _sumstep,
					QT* sqsum, size_t _sqsumstep, ST* tilted, size_t _tiltedstep,
					Size size, int cn );

	#define DEF_INTEGRAL_FUNC(suffix, T, ST, QT) \
	static void integralMultiImage_##suffix( T* src, size_t srcstep, ST* sum, size_t sumstep, QT* sqsum, size_t sqsumstep, \
								  ST* tilted, size_t tiltedstep, Size size, int cn ) \
	{ integralMultiImage_(src, srcstep, sum, sumstep, sqsum, sqsumstep, tilted, tiltedstep, size, cn); }

	DEF_INTEGRAL_FUNC(8u32s, uchar, int, double)
	DEF_INTEGRAL_FUNC(8u32f, uchar, float, double)
	DEF_INTEGRAL_FUNC(8u64f, uchar, double, double)
	DEF_INTEGRAL_FUNC(32f, float, float, double)
	DEF_INTEGRAL_FUNC(32f64f, float, double, double)
	DEF_INTEGRAL_FUNC(64f, double, double, double)

	typedef void (*IntegralFunc)(const uchar* src, size_t srcstep, uchar* sum, size_t sumstep,
								 uchar* sqsum, size_t sqsumstep, uchar* tilted, size_t tstep,
								 Size size, int cn );
}

#define GET_OPTIMIZED(func) (func)

typedef int sumtype;
typedef double sqsumtype;

typedef struct CvHidHaarFeature
{
    struct
    {
        sumtype *p0, *p1, *p2, *p3;
        float weight;
    }
    rect[CV_HAAR_FEATURE_MAX];
} CvHidHaarFeature;


typedef struct CvHidHaarTreeNode
{
    CvHidHaarFeature feature;
    float threshold;
    int left;
    int right;
} CvHidHaarTreeNode;


typedef struct CvHidHaarClassifier
{
    int count;
    //CvHaarFeature* orig_feature;
    CvHidHaarTreeNode* node;
    float* alpha;
} CvHidHaarClassifier;


typedef struct CvHidHaarStageClassifier
{
    int  count;
    float threshold;
    CvHidHaarClassifier* classifier;
    int two_rects;

    struct CvHidHaarStageClassifier* next;
    struct CvHidHaarStageClassifier* child;
    struct CvHidHaarStageClassifier* parent;
} CvHidHaarStageClassifier;


typedef struct CvHidHaarClassifierCascade
{
    int  count;
    int  isStumpBased;
    int  has_tilted_features;
    int  is_tree;
    double inv_window_area;
    CvMat sum, sqsum, tilted;
    CvHidHaarStageClassifier* stage_classifier;
    sqsumtype *pq0, *pq1, *pq2, *pq3;
    sumtype *p0, *p1, *p2, *p3;

    void** ipp_stages;
} CvHidHaarClassifierCascade;

const int icv_object_win_border = 1;
const float icv_stage_threshold_bias = 0.0001f;

CvSeq* cvHaarDetectObjectsMultiCameraForROC( const CvArr* image,
                     CvHaarClassifierCascade* cascade, CvMemStorage* storage,
                     std::vector<int>& rejectLevels, std::vector<double>& levelWeightds,
                     double scale_factor=1.1,
                     int min_neighbors=3, int flags=0,
                     CvSize min_size=cvSize(0,0), CvSize max_size=cvSize(0,0),
                     bool outputRejectLevels=false );

static CvHidHaarClassifierCascade*
icvCreateHidHaarClassifierCascade( CvHaarClassifierCascade* cascade );

double icvEvalHidHaarClassifier( CvHidHaarClassifier* classifier,
                                 double variance_norm_factor,
                                 size_t p_offset );

static int
cvRunHaarClassifierCascadeSum( const CvHaarClassifierCascade* _cascade,
                               CvPoint pt, double& stage_sum, int start_stage );
void
cvIntegralMultiImage( const CvArr* image, CvArr* sumImage,
            CvArr* sumSqImage, CvArr* tiltedSumImage );