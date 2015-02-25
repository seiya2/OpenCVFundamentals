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
			CvSeq* _objects = cvHaarDetectObjectsMultiCameraForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
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
		detectMultiScaleMultiCamera( image, objects, fakeLevels, fakeWeights, scaleFactor,
			minNeighbors, flags, minObjectSize, maxObjectSize, false );
	}

	HaarDetectObjects_ScaleImage_Invoker::HaarDetectObjects_ScaleImage_Invoker( const CvHaarClassifierCascade* _cascade,
											  int _stripSize, double _factor,
											  const Mat& _sum1, const Mat& _sqsum1, Mat* _norm1,
											  Mat* _mask1, Rect _equRect, std::vector<Rect>& _vec,
											  std::vector<int>& _levels, std::vector<double>& _weights,
											  bool _outputLevels, Mutex *_mtx )
	{
		cascade = _cascade;
		stripSize = _stripSize;
		factor = _factor;
		sum1 = _sum1;
		sqsum1 = _sqsum1;
		norm1 = _norm1;
		mask1 = _mask1;
		equRect = _equRect;
		vec = &_vec;
		rejectLevels = _outputLevels ? &_levels : 0;
		levelWeights = _outputLevels ? &_weights : 0;
		mtx = _mtx;
	}

	void HaarDetectObjects_ScaleImage_Invoker::operator()( const Range& range ) const
	{
		Size winSize0 = cascade->orig_window_size;
		Size winSize(cvRound(winSize0.width*factor), cvRound(winSize0.height*factor));
		int y1 = range.start*stripSize, y2 = min(range.end*stripSize, sum1.rows - 1 - winSize0.height);

		if (y2 <= y1 || sum1.cols <= 1 + winSize0.width)
			return;

		Size ssz(sum1.cols - 1 - winSize0.width, y2 - y1);
		int x, y, ystep = factor > 2 ? 1 : 2;

#ifdef HAVE_IPP
		if( cascade->hid_cascade->ipp_stages )
		{
			IppiRect iequRect = {equRect.x, equRect.y, equRect.width, equRect.height};
			ippiRectStdDev_32f_C1R(sum1.ptr<float>(y1), sum1.step,
									sqsum1.ptr<double>(y1), sqsum1.step,
									norm1->ptr<float>(y1), norm1->step,
									ippiSize(ssz.width, ssz.height), iequRect );

			int positive = (ssz.width/ystep)*((ssz.height + ystep-1)/ystep);

			if( ystep == 1 )
				(*mask1) = Scalar::all(1);
			else
				for( y = y1; y < y2; y++ )
				{
					uchar* mask1row = mask1->ptr(y);
					memset( mask1row, 0, ssz.width );

					if( y % ystep == 0 )
						for( x = 0; x < ssz.width; x += ystep )
							mask1row[x] = (uchar)1;
				}

			for( int j = 0; j < cascade->count; j++ )
			{
				if( ippiApplyHaarClassifier_32f_C1R(
							sum1.ptr<float>(y1), sum1.step,
							norm1->ptr<float>(y1), norm1->step,
							mask1->ptr<uchar>(y1), mask1->step,
							ippiSize(ssz.width, ssz.height), &positive,
							cascade->hid_cascade->stage_classifier[j].threshold,
							(IppiHaarClassifier_32f*)cascade->hid_cascade->ipp_stages[j]) < 0 )
					positive = 0;
				if( positive <= 0 )
					break;
			}

			if( positive > 0 )
				for( y = y1; y < y2; y += ystep )
				{
					uchar* mask1row = mask1->ptr(y);
					for( x = 0; x < ssz.width; x += ystep )
						if( mask1row[x] != 0 )
						{
							mtx->lock();
							vec->push_back(Rect(cvRound(x*factor), cvRound(y*factor),
												winSize.width, winSize.height));
							mtx->unlock();
							if( --positive == 0 )
								break;
						}
					if( positive == 0 )
						break;
				}
		}
		else
#endif // IPP
			for( y = y1; y < y2; y += ystep )
				for( x = 0; x < ssz.width; x += ystep )
				{
					double gypWeight;
					int result = cvRunHaarClassifierCascadeSum( cascade, cvPoint(x,y), gypWeight, 0 );
					if( rejectLevels )
					{
						if( result == 1 )
							result = -1*cascade->count;
						if( cascade->count + result < 4 )
						{
							mtx->lock();
							vec->push_back(Rect(cvRound(x*factor), cvRound(y*factor),
											winSize.width, winSize.height));
							rejectLevels->push_back(-result);
							levelWeights->push_back(gypWeight);
							mtx->unlock();
						}
					}
					else
					{
						if( result > 0 )
						{
							mtx->lock();
							vec->push_back(Rect(cvRound(x*factor), cvRound(y*factor),
											winSize.width, winSize.height));
							mtx->unlock();
						}
					}
				}
	}


	HaarDetectObjects_ScaleCascade_Invoker::HaarDetectObjects_ScaleCascade_Invoker( const CvHaarClassifierCascade* _cascade,
											Size _winsize, const Range& _xrange, double _ystep,
											size_t _sumstep, const int** _p, const int** _pq,
											std::vector<Rect>& _vec, Mutex* _mtx )
	{
		cascade = _cascade;
		winsize = _winsize;
		xrange = _xrange;
		ystep = _ystep;
		sumstep = _sumstep;
		p = _p; pq = _pq;
		vec = &_vec;
		mtx = _mtx;
	}

	void HaarDetectObjects_ScaleCascade_Invoker::operator()( const Range& range ) const
	{
		int iy, startY = range.start, endY = range.end;
		const int *p0 = p[0], *p1 = p[1], *p2 = p[2], *p3 = p[3];
		const int *pq0 = pq[0], *pq1 = pq[1], *pq2 = pq[2], *pq3 = pq[3];
		bool doCannyPruning = p0 != 0;
		int sstep = (int)(sumstep/sizeof(p0[0]));

		for( iy = startY; iy < endY; iy++ )
		{
			int ix, y = cvRound(iy*ystep), ixstep = 1;
			for( ix = xrange.start; ix < xrange.end; ix += ixstep )
			{
				int x = cvRound(ix*ystep); // it should really be ystep, not ixstep

				if( doCannyPruning )
				{
					int offset = y*sstep + x;
					int s = p0[offset] - p1[offset] - p2[offset] + p3[offset];
					int sq = pq0[offset] - pq1[offset] - pq2[offset] + pq3[offset];
					if( s < 100 || sq < 20 )
					{
						ixstep = 2;
						continue;
					}
				}

				int result = cvRunHaarClassifierCascade( cascade, cvPoint(x, y), 0 );
				if( result > 0 )
				{
					mtx->lock();
					vec->push_back(Rect(x, y, winsize.width, winsize.height));
					mtx->unlock();
				}
				ixstep = result != 0 ? 1 : 2;
			}
		}
	}

	template<typename T, typename ST, typename QT>
	void integralMultiImage_( const T* src, size_t _srcstep, ST* sum, size_t _sumstep,
					QT* sqsum, size_t _sqsumstep, ST* tilted, size_t _tiltedstep,
					Size size, int cn )
	{
		int x, y, k;

		int srcstep = (int)(_srcstep/sizeof(T));
		int sumstep = (int)(_sumstep/sizeof(ST));
		int tiltedstep = (int)(_tiltedstep/sizeof(ST));
		int sqsumstep = (int)(_sqsumstep/sizeof(QT));

		size.width *= cn;

		memset( sum, 0, (size.width+cn)*sizeof(sum[0]));
		sum += sumstep + cn;

		if( sqsum )
		{
			memset( sqsum, 0, (size.width+cn)*sizeof(sqsum[0]));
			sqsum += sqsumstep + cn;
		}

		if( tilted )
		{
			memset( tilted, 0, (size.width+cn)*sizeof(tilted[0]));
			tilted += tiltedstep + cn;
		}

		if( sqsum == 0 && tilted == 0 )
		{
			for( y = 0; y < size.height; y++, src += srcstep - cn, sum += sumstep - cn )
			{
				for( k = 0; k < cn; k++, src++, sum++ )
				{
					ST s = sum[-cn] = 0;
					for( x = 0; x < size.width; x += cn )
					{
						s += src[x];
						sum[x] = sum[x - sumstep] + s;
					}
				}
			}
		}
		else if( tilted == 0 )
		{
			for( y = 0; y < size.height; y++, src += srcstep - cn,
							sum += sumstep - cn, sqsum += sqsumstep - cn )
			{
				for( k = 0; k < cn; k++, src++, sum++, sqsum++ )
				{
					ST s = sum[-cn] = 0;
					QT sq = sqsum[-cn] = 0;
					for( x = 0; x < size.width; x += cn )
					{
						T it = src[x];
						s += it;
						sq += (QT)it*it;
						ST t = sum[x - sumstep] + s;
						QT tq = sqsum[x - sqsumstep] + sq;
						sum[x] = t;
						sqsum[x] = tq;
					}
				}
			}
		}
		else
		{
			AutoBuffer<ST> _buf(size.width+cn);
			ST* buf = _buf;
			ST s;
			QT sq;
			for( k = 0; k < cn; k++, src++, sum++, tilted++, buf++ )
			{
				sum[-cn] = tilted[-cn] = 0;

				for( x = 0, s = 0, sq = 0; x < size.width; x += cn )
				{
					T it = src[x];
					buf[x] = tilted[x] = it;
					s += it;
					sq += (QT)it*it;
					sum[x] = s;
					if( sqsum )
						sqsum[x] = sq;
				}

				if( size.width == cn )
					buf[cn] = 0;

				if( sqsum )
				{
					sqsum[-cn] = 0;
					sqsum++;
				}
			}

			for( y = 1; y < size.height; y++ )
			{
				src += srcstep - cn;
				sum += sumstep - cn;
				tilted += tiltedstep - cn;
				buf += -cn;

				if( sqsum )
					sqsum += sqsumstep - cn;

				for( k = 0; k < cn; k++, src++, sum++, tilted++, buf++ )
				{
					T it = src[0];
					ST t0 = s = it;
					QT tq0 = sq = (QT)it*it;

					sum[-cn] = 0;
					if( sqsum )
						sqsum[-cn] = 0;
					tilted[-cn] = tilted[-tiltedstep];

					sum[0] = sum[-sumstep] + t0;
					if( sqsum )
						sqsum[0] = sqsum[-sqsumstep] + tq0;
					tilted[0] = tilted[-tiltedstep] + t0 + buf[cn];

					for( x = cn; x < size.width - cn; x += cn )
					{
						ST t1 = buf[x];
						buf[x - cn] = t1 + t0;
						t0 = it = src[x];
						tq0 = (QT)it*it;
						s += t0;
						sq += tq0;
						sum[x] = sum[x - sumstep] + s;
						if( sqsum )
							sqsum[x] = sqsum[x - sqsumstep] + sq;
						t1 += buf[x + cn] + t0 + tilted[x - tiltedstep - cn];
						tilted[x] = t1;
					}

					if( size.width > cn )
					{
						ST t1 = buf[x];
						buf[x - cn] = t1 + t0;
						t0 = it = src[x];
						tq0 = (QT)it*it;
						s += t0;
						sq += tq0;
						sum[x] = sum[x - sumstep] + s;
						if( sqsum )
							sqsum[x] = sqsum[x - sqsumstep] + sq;
						tilted[x] = t0 + t1 + tilted[x - tiltedstep - cn];
						buf[x] = t0;
					}

					if( sqsum )
						sqsum++;
				}
			}
		}
	}

	void integralMultiImage( InputArray _src, OutputArray _sum, OutputArray _sqsum, OutputArray _tilted, int sdepth )
	{
		Mat src = _src.getMat(), sum, sqsum, tilted;
		int depth = src.depth(), cn = src.channels();
		Size isize(src.cols + 1, src.rows+1);

		if( sdepth <= 0 )
			sdepth = depth == CV_8U ? CV_32S : CV_64F;
		sdepth = CV_MAT_DEPTH(sdepth);

	#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
		if( ( depth == CV_8U ) && ( !_tilted.needed() ) )
		{
			if( sdepth == CV_32F )
			{
				if( cn == 1 )
				{
					IppiSize srcRoiSize = ippiSize( src.cols, src.rows );
					_sum.create( isize, CV_MAKETYPE( sdepth, cn ) );
					sum = _sum.getMat();
					if( _sqsum.needed() )
					{
						_sqsum.create( isize, CV_MAKETYPE( CV_64F, cn ) );
						sqsum = _sqsum.getMat();
						ippiSqrIntegral_8u32f64f_C1R( (const Ipp8u*)src.data, (int)src.step, (Ipp32f*)sum.data, (int)sum.step, (Ipp64f*)sqsum.data, (int)sqsum.step, srcRoiSize, 0, 0 );
					}
					else
					{
						ippiIntegral_8u32f_C1R( (const Ipp8u*)src.data, (int)src.step, (Ipp32f*)sum.data, (int)sum.step, srcRoiSize, 0 );
					}
					return;
				}
			}
			if( sdepth == CV_32S )
			{
				if( cn == 1 )
				{
					IppiSize srcRoiSize = ippiSize( src.cols, src.rows );
					_sum.create( isize, CV_MAKETYPE( sdepth, cn ) );
					sum = _sum.getMat();
					if( _sqsum.needed() )
					{
						_sqsum.create( isize, CV_MAKETYPE( CV_64F, cn ) );
						sqsum = _sqsum.getMat();
						ippiSqrIntegral_8u32s64f_C1R( (const Ipp8u*)src.data, (int)src.step, (Ipp32s*)sum.data, (int)sum.step, (Ipp64f*)sqsum.data, (int)sqsum.step, srcRoiSize, 0, 0 );
					}
					else
					{
						ippiIntegral_8u32s_C1R( (const Ipp8u*)src.data, (int)src.step, (Ipp32s*)sum.data, (int)sum.step, srcRoiSize, 0 );
					}
					return;
				}
			}
		}
	#endif

		_sum.create( isize, CV_MAKETYPE(sdepth, cn) );
		sum = _sum.getMat();

		if( _tilted.needed() )
		{
			_tilted.create( isize, CV_MAKETYPE(sdepth, cn) );
			tilted = _tilted.getMat();
		}

		if( _sqsum.needed() )
		{
			_sqsum.create( isize, CV_MAKETYPE(CV_64F, cn) );
			sqsum = _sqsum.getMat();
		}

		IntegralFunc func = 0;

		if( depth == CV_8U && sdepth == CV_32S )
			func = (IntegralFunc)GET_OPTIMIZED(integralMultiImage_8u32s);
		else if( depth == CV_8U && sdepth == CV_32F )
			func = (IntegralFunc)integralMultiImage_8u32f;
		else if( depth == CV_8U && sdepth == CV_64F )
			func = (IntegralFunc)integralMultiImage_8u64f;
		else if( depth == CV_32F && sdepth == CV_32F )
			func = (IntegralFunc)integralMultiImage_32f;
		else if( depth == CV_32F && sdepth == CV_64F )
			func = (IntegralFunc)integralMultiImage_32f64f;
		else if( depth == CV_64F && sdepth == CV_64F )
			func = (IntegralFunc)integralMultiImage_64f;
		else
			CV_Error( CV_StsUnsupportedFormat, "" );

		func( src.data, src.step, sum.data, sum.step, sqsum.data, sqsum.step,
			  tilted.data, tilted.step, src.size(), cn );
	}

	void integralMultiImage( InputArray src, OutputArray sum, int sdepth )
	{
		integralMultiImage( src, sum, noArray(), noArray(), sdepth );
	}

	void integralMultiImage( InputArray src, OutputArray sum, OutputArray sqsum, int sdepth )
	{
		integralMultiImage( src, sum, sqsum, noArray(), sdepth );
	}
}

#define sum_elem_ptr(sum,row,col)  \
    ((sumtype*)CV_MAT_ELEM_PTR_FAST((sum),(row),(col),sizeof(sumtype)))

#define sqsum_elem_ptr(sqsum,row,col)  \
    ((sqsumtype*)CV_MAT_ELEM_PTR_FAST((sqsum),(row),(col),sizeof(sqsumtype)))

#define calc_sum(rect,offset) \
    ((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

#define calc_sumf(rect,offset) \
    static_cast<float>((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])



static CvHidHaarClassifierCascade*
icvCreateHidHaarClassifierCascade( CvHaarClassifierCascade* cascade )
{
    CvRect* ipp_features = 0;
    float *ipp_weights = 0, *ipp_thresholds = 0, *ipp_val1 = 0, *ipp_val2 = 0;
    int* ipp_counts = 0;

    CvHidHaarClassifierCascade* out = 0;

    int i, j, k, l;
    int datasize;
    int total_classifiers = 0;
    int total_nodes = 0;
    char errorstr[1000];
    CvHidHaarClassifier* haar_classifier_ptr;
    CvHidHaarTreeNode* haar_node_ptr;
    CvSize orig_window_size;
    int has_tilted_features = 0;
    int max_count = 0;

    if( !CV_IS_HAAR_CLASSIFIER(cascade) )
        CV_Error( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

    if( cascade->hid_cascade )
        CV_Error( CV_StsError, "hid_cascade has been already created" );

    if( !cascade->stage_classifier )
        CV_Error( CV_StsNullPtr, "" );

    if( cascade->count <= 0 )
        CV_Error( CV_StsOutOfRange, "Negative number of cascade stages" );

    orig_window_size = cascade->orig_window_size;

    /* check input structure correctness and calculate total memory size needed for
       internal representation of the classifier cascade */
    for( i = 0; i < cascade->count; i++ )
    {
        CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;

        if( !stage_classifier->classifier ||
            stage_classifier->count <= 0 )
        {
            sprintf( errorstr, "header of the stage classifier #%d is invalid "
                     "(has null pointers or non-positive classfier count)", i );
            CV_Error( CV_StsError, errorstr );
        }

        max_count = MAX( max_count, stage_classifier->count );
        total_classifiers += stage_classifier->count;

        for( j = 0; j < stage_classifier->count; j++ )
        {
            CvHaarClassifier* classifier = stage_classifier->classifier + j;

            total_nodes += classifier->count;
            for( l = 0; l < classifier->count; l++ )
            {
                for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
                {
                    if( classifier->haar_feature[l].rect[k].r.width )
                    {
                        CvRect r = classifier->haar_feature[l].rect[k].r;
                        int tilted = classifier->haar_feature[l].tilted;
                        has_tilted_features |= tilted != 0;
                        if( r.width < 0 || r.height < 0 || r.y < 0 ||
                            r.x + r.width > orig_window_size.width
                            ||
                            (!tilted &&
                            (r.x < 0 || r.y + r.height > orig_window_size.height))
                            ||
                            (tilted && (r.x - r.height < 0 ||
                            r.y + r.width + r.height > orig_window_size.height)))
                        {
                            sprintf( errorstr, "rectangle #%d of the classifier #%d of "
                                     "the stage classifier #%d is not inside "
                                     "the reference (original) cascade window", k, j, i );
                            CV_Error( CV_StsNullPtr, errorstr );
                        }
                    }
                }
            }
        }
    }

    // this is an upper boundary for the whole hidden cascade size
    datasize = sizeof(CvHidHaarClassifierCascade) +
               sizeof(CvHidHaarStageClassifier)*cascade->count +
               sizeof(CvHidHaarClassifier) * total_classifiers +
               sizeof(CvHidHaarTreeNode) * total_nodes +
               sizeof(void*)*(total_nodes + total_classifiers);

    out = (CvHidHaarClassifierCascade*)cvAlloc( datasize );
    memset( out, 0, sizeof(*out) );

    /* init header */
    out->count = cascade->count;
    out->stage_classifier = (CvHidHaarStageClassifier*)(out + 1);
    haar_classifier_ptr = (CvHidHaarClassifier*)(out->stage_classifier + cascade->count);
    haar_node_ptr = (CvHidHaarTreeNode*)(haar_classifier_ptr + total_classifiers);

    out->isStumpBased = 1;
    out->has_tilted_features = has_tilted_features;
    out->is_tree = 0;

    /* initialize internal representation */
    for( i = 0; i < cascade->count; i++ )
    {
        CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;
        CvHidHaarStageClassifier* hid_stage_classifier = out->stage_classifier + i;

        hid_stage_classifier->count = stage_classifier->count;
        hid_stage_classifier->threshold = stage_classifier->threshold - icv_stage_threshold_bias;
        hid_stage_classifier->classifier = haar_classifier_ptr;
        hid_stage_classifier->two_rects = 1;
        haar_classifier_ptr += stage_classifier->count;

        hid_stage_classifier->parent = (stage_classifier->parent == -1)
            ? NULL : out->stage_classifier + stage_classifier->parent;
        hid_stage_classifier->next = (stage_classifier->next == -1)
            ? NULL : out->stage_classifier + stage_classifier->next;
        hid_stage_classifier->child = (stage_classifier->child == -1)
            ? NULL : out->stage_classifier + stage_classifier->child;

        out->is_tree |= hid_stage_classifier->next != NULL;

        for( j = 0; j < stage_classifier->count; j++ )
        {
            CvHaarClassifier* classifier = stage_classifier->classifier + j;
            CvHidHaarClassifier* hid_classifier = hid_stage_classifier->classifier + j;
            int node_count = classifier->count;
            float* alpha_ptr = (float*)(haar_node_ptr + node_count);

            hid_classifier->count = node_count;
            hid_classifier->node = haar_node_ptr;
            hid_classifier->alpha = alpha_ptr;

            for( l = 0; l < node_count; l++ )
            {
                CvHidHaarTreeNode* node = hid_classifier->node + l;
                CvHaarFeature* feature = classifier->haar_feature + l;
                memset( node, -1, sizeof(*node) );
                node->threshold = classifier->threshold[l];
                node->left = classifier->left[l];
                node->right = classifier->right[l];

                if( fabs(feature->rect[2].weight) < DBL_EPSILON ||
                    feature->rect[2].r.width == 0 ||
                    feature->rect[2].r.height == 0 )
                    memset( &(node->feature.rect[2]), 0, sizeof(node->feature.rect[2]) );
                else
                    hid_stage_classifier->two_rects = 0;
            }

            memcpy( alpha_ptr, classifier->alpha, (node_count+1)*sizeof(alpha_ptr[0]));
            haar_node_ptr =
                (CvHidHaarTreeNode*)cvAlignPtr(alpha_ptr+node_count+1, sizeof(void*));

            out->isStumpBased &= node_count == 1;
        }
    }
/*
#ifdef HAVE_IPP
    int can_use_ipp = !out->has_tilted_features && !out->is_tree && out->isStumpBased;

    if( can_use_ipp )
    {
        int ipp_datasize = cascade->count*sizeof(out->ipp_stages[0]);
        float ipp_weight_scale=(float)(1./((orig_window_size.width-icv_object_win_border*2)*
            (orig_window_size.height-icv_object_win_border*2)));

        out->ipp_stages = (void**)cvAlloc( ipp_datasize );
        memset( out->ipp_stages, 0, ipp_datasize );

        ipp_features = (CvRect*)cvAlloc( max_count*3*sizeof(ipp_features[0]) );
        ipp_weights = (float*)cvAlloc( max_count*3*sizeof(ipp_weights[0]) );
        ipp_thresholds = (float*)cvAlloc( max_count*sizeof(ipp_thresholds[0]) );
        ipp_val1 = (float*)cvAlloc( max_count*sizeof(ipp_val1[0]) );
        ipp_val2 = (float*)cvAlloc( max_count*sizeof(ipp_val2[0]) );
        ipp_counts = (int*)cvAlloc( max_count*sizeof(ipp_counts[0]) );

        for( i = 0; i < cascade->count; i++ )
        {
            CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;
            for( j = 0, k = 0; j < stage_classifier->count; j++ )
            {
                CvHaarClassifier* classifier = stage_classifier->classifier + j;
                int rect_count = 2 + (classifier->haar_feature->rect[2].r.width != 0);

                ipp_thresholds[j] = classifier->threshold[0];
                ipp_val1[j] = classifier->alpha[0];
                ipp_val2[j] = classifier->alpha[1];
                ipp_counts[j] = rect_count;

                for( l = 0; l < rect_count; l++, k++ )
                {
                    ipp_features[k] = classifier->haar_feature->rect[l].r;
                    //ipp_features[k].y = orig_window_size.height - ipp_features[k].y - ipp_features[k].height;
                    ipp_weights[k] = classifier->haar_feature->rect[l].weight*ipp_weight_scale;
                }
            }

            if( ippiHaarClassifierInitAlloc_32f( (IppiHaarClassifier_32f**)&out->ipp_stages[i],
                (const IppiRect*)ipp_features, ipp_weights, ipp_thresholds,
                ipp_val1, ipp_val2, ipp_counts, stage_classifier->count ) < 0 )
                break;
        }

        if( i < cascade->count )
        {
            for( j = 0; j < i; j++ )
                if( out->ipp_stages[i] )
                    ippiHaarClassifierFree_32f( (IppiHaarClassifier_32f*)out->ipp_stages[i] );
            cvFree( &out->ipp_stages );
        }
    }
#endif
*/
    cascade->hid_cascade = out;
    assert( (char*)haar_node_ptr - (char*)out <= datasize );

    cvFree( &ipp_features );
    cvFree( &ipp_weights );
    cvFree( &ipp_thresholds );
    cvFree( &ipp_val1 );
    cvFree( &ipp_val2 );
    cvFree( &ipp_counts );

    return out;
}

static int
cvRunHaarClassifierCascadeSum( const CvHaarClassifierCascade* _cascade,
                               CvPoint pt, double& stage_sum, int start_stage )
{
#ifdef CV_HAAR_USE_AVX
    bool haveAVX = false;
    if(cv::checkHardwareSupport(CV_CPU_AVX))
    if(__xgetbv()&0x6)// Check if the OS will save the YMM registers
       haveAVX = true;
#else
#  ifdef CV_HAAR_USE_SSE
    bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#  endif
#endif

    int p_offset, pq_offset;
    int i, j;
    double mean, variance_norm_factor;
    CvHidHaarClassifierCascade* cascade;

    if( !CV_IS_HAAR_CLASSIFIER(_cascade) )
        CV_Error( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid cascade pointer" );

    cascade = _cascade->hid_cascade;
    if( !cascade )
        CV_Error( CV_StsNullPtr, "Hidden cascade has not been created.\n"
            "Use cvSetImagesForHaarClassifierCascade" );

    if( pt.x < 0 || pt.y < 0 ||
        pt.x + _cascade->real_window_size.width >= cascade->sum.width ||
        pt.y + _cascade->real_window_size.height >= cascade->sum.height )
        return -1;

    p_offset = pt.y * (cascade->sum.step/sizeof(sumtype)) + pt.x;
    pq_offset = pt.y * (cascade->sqsum.step/sizeof(sqsumtype)) + pt.x;
    mean = calc_sum(*cascade,p_offset)*cascade->inv_window_area;
    variance_norm_factor = cascade->pq0[pq_offset] - cascade->pq1[pq_offset] -
                           cascade->pq2[pq_offset] + cascade->pq3[pq_offset];
    variance_norm_factor = variance_norm_factor*cascade->inv_window_area - mean*mean;
    if( variance_norm_factor >= 0. )
        variance_norm_factor = sqrt(variance_norm_factor);
    else
        variance_norm_factor = 1.;

    if( cascade->is_tree )
    {
        CvHidHaarStageClassifier* ptr = cascade->stage_classifier;
        assert( start_stage == 0 );

        while( ptr )
        {
            stage_sum = 0.0;
            j = 0;

#ifdef CV_HAAR_USE_AVX
            if(haveAVX)
            {
                for( ; j <= ptr->count - 8; j += 8 )
                {
                    stage_sum += icvEvalHidHaarClassifierAVX(
                        ptr->classifier + j,
                        variance_norm_factor, p_offset );
                }
            }
#endif
            for( ; j < ptr->count; j++ )
            {
                stage_sum += icvEvalHidHaarClassifier( ptr->classifier + j, variance_norm_factor, p_offset );
            }

            if( stage_sum >= ptr->threshold )
            {
                ptr = ptr->child;
            }
            else
            {
                while( ptr && ptr->next == NULL ) ptr = ptr->parent;
                if( ptr == NULL )
                    return 0;
                ptr = ptr->next;
            }
        }
    }
    else if( cascade->isStumpBased )
    {
#ifdef CV_HAAR_USE_AVX
        if(haveAVX)
        {
            CvHidHaarClassifier* classifiers[8];
            CvHidHaarTreeNode* nodes[8];
            for( i = start_stage; i < cascade->count; i++ )
            {
                stage_sum = 0.0;
                j = 0;
                float CV_DECL_ALIGNED(32) buf[8];
                if( cascade->stage_classifier[i].two_rects )
                {
                    for( ; j <= cascade->stage_classifier[i].count - 8; j += 8 )
                    {
                        classifiers[0] = cascade->stage_classifier[i].classifier + j;
                        nodes[0] = classifiers[0]->node;
                        classifiers[1] = cascade->stage_classifier[i].classifier + j + 1;
                        nodes[1] = classifiers[1]->node;
                        classifiers[2] = cascade->stage_classifier[i].classifier + j + 2;
                        nodes[2] = classifiers[2]->node;
                        classifiers[3] = cascade->stage_classifier[i].classifier + j + 3;
                        nodes[3] = classifiers[3]->node;
                        classifiers[4] = cascade->stage_classifier[i].classifier + j + 4;
                        nodes[4] = classifiers[4]->node;
                        classifiers[5] = cascade->stage_classifier[i].classifier + j + 5;
                        nodes[5] = classifiers[5]->node;
                        classifiers[6] = cascade->stage_classifier[i].classifier + j + 6;
                        nodes[6] = classifiers[6]->node;
                        classifiers[7] = cascade->stage_classifier[i].classifier + j + 7;
                        nodes[7] = classifiers[7]->node;

                        __m256 t = _mm256_set1_ps(static_cast<float>(variance_norm_factor));
                        t = _mm256_mul_ps(t, _mm256_set_ps(nodes[7]->threshold,
                                                           nodes[6]->threshold,
                                                           nodes[5]->threshold,
                                                           nodes[4]->threshold,
                                                           nodes[3]->threshold,
                                                           nodes[2]->threshold,
                                                           nodes[1]->threshold,
                                                           nodes[0]->threshold));

                        __m256 offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[6]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[5]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[4]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[3]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[2]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[1]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[0]->feature.rect[0], p_offset));

                        __m256 weight = _mm256_set_ps(nodes[7]->feature.rect[0].weight,
                                                      nodes[6]->feature.rect[0].weight,
                                                      nodes[5]->feature.rect[0].weight,
                                                      nodes[4]->feature.rect[0].weight,
                                                      nodes[3]->feature.rect[0].weight,
                                                      nodes[2]->feature.rect[0].weight,
                                                      nodes[1]->feature.rect[0].weight,
                                                      nodes[0]->feature.rect[0].weight);

                        __m256 sum = _mm256_mul_ps(offset, weight);

                        offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[6]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[5]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[4]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[3]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[2]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[1]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[0]->feature.rect[1], p_offset));

                        weight = _mm256_set_ps(nodes[7]->feature.rect[1].weight,
                                               nodes[6]->feature.rect[1].weight,
                                               nodes[5]->feature.rect[1].weight,
                                               nodes[4]->feature.rect[1].weight,
                                               nodes[3]->feature.rect[1].weight,
                                               nodes[2]->feature.rect[1].weight,
                                               nodes[1]->feature.rect[1].weight,
                                               nodes[0]->feature.rect[1].weight);

                        sum = _mm256_add_ps(sum, _mm256_mul_ps(offset,weight));

                        __m256 alpha0 = _mm256_set_ps(classifiers[7]->alpha[0],
                                                      classifiers[6]->alpha[0],
                                                      classifiers[5]->alpha[0],
                                                      classifiers[4]->alpha[0],
                                                      classifiers[3]->alpha[0],
                                                      classifiers[2]->alpha[0],
                                                      classifiers[1]->alpha[0],
                                                      classifiers[0]->alpha[0]);
                        __m256 alpha1 = _mm256_set_ps(classifiers[7]->alpha[1],
                                                      classifiers[6]->alpha[1],
                                                      classifiers[5]->alpha[1],
                                                      classifiers[4]->alpha[1],
                                                      classifiers[3]->alpha[1],
                                                      classifiers[2]->alpha[1],
                                                      classifiers[1]->alpha[1],
                                                      classifiers[0]->alpha[1]);

                        _mm256_store_ps(buf, _mm256_blendv_ps(alpha0, alpha1, _mm256_cmp_ps(t, sum, _CMP_LE_OQ)));
                        stage_sum += (buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7]);
                    }

                    for( ; j < cascade->stage_classifier[i].count; j++ )
                    {
                        CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                        CvHidHaarTreeNode* node = classifier->node;

                        double t = node->threshold*variance_norm_factor;
                        double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
                        sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
                        stage_sum += classifier->alpha[sum >= t];
                    }
                }
                else
                {
                    for( ; j <= (cascade->stage_classifier[i].count)-8; j+=8 )
                    {
                        float  CV_DECL_ALIGNED(32) tmp[8] = {0,0,0,0,0,0,0,0};

                        classifiers[0] = cascade->stage_classifier[i].classifier + j;
                        nodes[0] = classifiers[0]->node;
                        classifiers[1] = cascade->stage_classifier[i].classifier + j + 1;
                        nodes[1] = classifiers[1]->node;
                        classifiers[2] = cascade->stage_classifier[i].classifier + j + 2;
                        nodes[2] = classifiers[2]->node;
                        classifiers[3] = cascade->stage_classifier[i].classifier + j + 3;
                        nodes[3] = classifiers[3]->node;
                        classifiers[4] = cascade->stage_classifier[i].classifier + j + 4;
                        nodes[4] = classifiers[4]->node;
                        classifiers[5] = cascade->stage_classifier[i].classifier + j + 5;
                        nodes[5] = classifiers[5]->node;
                        classifiers[6] = cascade->stage_classifier[i].classifier + j + 6;
                        nodes[6] = classifiers[6]->node;
                        classifiers[7] = cascade->stage_classifier[i].classifier + j + 7;
                        nodes[7] = classifiers[7]->node;

                        __m256 t = _mm256_set1_ps(static_cast<float>(variance_norm_factor));

                        t = _mm256_mul_ps(t, _mm256_set_ps(nodes[7]->threshold,
                                                           nodes[6]->threshold,
                                                           nodes[5]->threshold,
                                                           nodes[4]->threshold,
                                                           nodes[3]->threshold,
                                                           nodes[2]->threshold,
                                                           nodes[1]->threshold,
                                                           nodes[0]->threshold));

                        __m256 offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[6]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[5]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[4]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[3]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[2]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[1]->feature.rect[0], p_offset),
                                                      calc_sumf(nodes[0]->feature.rect[0], p_offset));

                        __m256 weight = _mm256_set_ps(nodes[7]->feature.rect[0].weight,
                                                      nodes[6]->feature.rect[0].weight,
                                                      nodes[5]->feature.rect[0].weight,
                                                      nodes[4]->feature.rect[0].weight,
                                                      nodes[3]->feature.rect[0].weight,
                                                      nodes[2]->feature.rect[0].weight,
                                                      nodes[1]->feature.rect[0].weight,
                                                      nodes[0]->feature.rect[0].weight);

                        __m256 sum = _mm256_mul_ps(offset, weight);

                        offset = _mm256_set_ps(calc_sumf(nodes[7]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[6]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[5]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[4]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[3]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[2]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[1]->feature.rect[1], p_offset),
                                               calc_sumf(nodes[0]->feature.rect[1], p_offset));

                        weight = _mm256_set_ps(nodes[7]->feature.rect[1].weight,
                                               nodes[6]->feature.rect[1].weight,
                                               nodes[5]->feature.rect[1].weight,
                                               nodes[4]->feature.rect[1].weight,
                                               nodes[3]->feature.rect[1].weight,
                                               nodes[2]->feature.rect[1].weight,
                                               nodes[1]->feature.rect[1].weight,
                                               nodes[0]->feature.rect[1].weight);

                        sum = _mm256_add_ps(sum, _mm256_mul_ps(offset, weight));

                        if( nodes[0]->feature.rect[2].p0 )
                            tmp[0] = calc_sumf(nodes[0]->feature.rect[2],p_offset) * nodes[0]->feature.rect[2].weight;
                        if( nodes[1]->feature.rect[2].p0 )
                            tmp[1] = calc_sumf(nodes[1]->feature.rect[2],p_offset) * nodes[1]->feature.rect[2].weight;
                        if( nodes[2]->feature.rect[2].p0 )
                            tmp[2] = calc_sumf(nodes[2]->feature.rect[2],p_offset) * nodes[2]->feature.rect[2].weight;
                        if( nodes[3]->feature.rect[2].p0 )
                            tmp[3] = calc_sumf(nodes[3]->feature.rect[2],p_offset) * nodes[3]->feature.rect[2].weight;
                        if( nodes[4]->feature.rect[2].p0 )
                            tmp[4] = calc_sumf(nodes[4]->feature.rect[2],p_offset) * nodes[4]->feature.rect[2].weight;
                        if( nodes[5]->feature.rect[2].p0 )
                            tmp[5] = calc_sumf(nodes[5]->feature.rect[2],p_offset) * nodes[5]->feature.rect[2].weight;
                        if( nodes[6]->feature.rect[2].p0 )
                            tmp[6] = calc_sumf(nodes[6]->feature.rect[2],p_offset) * nodes[6]->feature.rect[2].weight;
                        if( nodes[7]->feature.rect[2].p0 )
                            tmp[7] = calc_sumf(nodes[7]->feature.rect[2],p_offset) * nodes[7]->feature.rect[2].weight;

                        sum = _mm256_add_ps(sum, _mm256_load_ps(tmp));

                        __m256 alpha0 = _mm256_set_ps(classifiers[7]->alpha[0],
                                                      classifiers[6]->alpha[0],
                                                      classifiers[5]->alpha[0],
                                                      classifiers[4]->alpha[0],
                                                      classifiers[3]->alpha[0],
                                                      classifiers[2]->alpha[0],
                                                      classifiers[1]->alpha[0],
                                                      classifiers[0]->alpha[0]);
                        __m256 alpha1 = _mm256_set_ps(classifiers[7]->alpha[1],
                                                      classifiers[6]->alpha[1],
                                                      classifiers[5]->alpha[1],
                                                      classifiers[4]->alpha[1],
                                                      classifiers[3]->alpha[1],
                                                      classifiers[2]->alpha[1],
                                                      classifiers[1]->alpha[1],
                                                      classifiers[0]->alpha[1]);

                        __m256 outBuf = _mm256_blendv_ps(alpha0, alpha1, _mm256_cmp_ps(t, sum, _CMP_LE_OQ ));
                        outBuf = _mm256_hadd_ps(outBuf, outBuf);
                        outBuf = _mm256_hadd_ps(outBuf, outBuf);
                        _mm256_store_ps(buf, outBuf);
                        stage_sum += (buf[0] + buf[4]);
                    }

                    for( ; j < cascade->stage_classifier[i].count; j++ )
                    {
                        CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                        CvHidHaarTreeNode* node = classifier->node;

                        double t = node->threshold*variance_norm_factor;
                        double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
                        sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
                        if( node->feature.rect[2].p0 )
                            sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;
                        stage_sum += classifier->alpha[sum >= t];
                    }
                }
                if( stage_sum < cascade->stage_classifier[i].threshold )
                    return -i;
            }
        }
        else
#elif defined CV_HAAR_USE_SSE //old SSE optimization
        if(haveSSE2)
        {
            for( i = start_stage; i < cascade->count; i++ )
            {
                __m128d vstage_sum = _mm_setzero_pd();
                if( cascade->stage_classifier[i].two_rects )
                {
                    for( j = 0; j < cascade->stage_classifier[i].count; j++ )
                    {
                        CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                        CvHidHaarTreeNode* node = classifier->node;

                        // ayasin - NHM perf optim. Avoid use of costly flaky jcc
                        __m128d t = _mm_set_sd(node->threshold*variance_norm_factor);
                        __m128d a = _mm_set_sd(classifier->alpha[0]);
                        __m128d b = _mm_set_sd(classifier->alpha[1]);
                        __m128d sum = _mm_set_sd(calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight +
                                                 calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight);
                        t = _mm_cmpgt_sd(t, sum);
                        vstage_sum = _mm_add_sd(vstage_sum, _mm_blendv_pd(b, a, t));
                    }
                }
                else
                {
                    for( j = 0; j < cascade->stage_classifier[i].count; j++ )
                    {
                        CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                        CvHidHaarTreeNode* node = classifier->node;
                        // ayasin - NHM perf optim. Avoid use of costly flaky jcc
                        __m128d t = _mm_set_sd(node->threshold*variance_norm_factor);
                        __m128d a = _mm_set_sd(classifier->alpha[0]);
                        __m128d b = _mm_set_sd(classifier->alpha[1]);
                        double _sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
                        _sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
                        if( node->feature.rect[2].p0 )
                            _sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;
                        __m128d sum = _mm_set_sd(_sum);

                        t = _mm_cmpgt_sd(t, sum);
                        vstage_sum = _mm_add_sd(vstage_sum, _mm_blendv_pd(b, a, t));
                    }
                }
                __m128d i_threshold = _mm_set1_pd(cascade->stage_classifier[i].threshold);
                if( _mm_comilt_sd(vstage_sum, i_threshold) )
                    return -i;
            }
        }
        else
#endif // AVX or SSE
        {
            for( i = start_stage; i < cascade->count; i++ )
            {
                stage_sum = 0.0;
                if( cascade->stage_classifier[i].two_rects )
                {
                    for( j = 0; j < cascade->stage_classifier[i].count; j++ )
                    {
                        CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                        CvHidHaarTreeNode* node = classifier->node;
                        double t = node->threshold*variance_norm_factor;
                        double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
                        sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
                        stage_sum += classifier->alpha[sum >= t];
                    }
                }
                else
                {
                    for( j = 0; j < cascade->stage_classifier[i].count; j++ )
                    {
                        CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
                        CvHidHaarTreeNode* node = classifier->node;
                        double t = node->threshold*variance_norm_factor;
                        double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
                        sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
                        if( node->feature.rect[2].p0 )
                            sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;
                        stage_sum += classifier->alpha[sum >= t];
                    }
                }
                if( stage_sum < cascade->stage_classifier[i].threshold )
                    return -i;
            }
        }
    }
    else
    {
        for( i = start_stage; i < cascade->count; i++ )
        {
            stage_sum = 0.0;
            int k = 0;

#ifdef CV_HAAR_USE_AVX
            if(haveAVX)
            {
                for( ; k < cascade->stage_classifier[i].count - 8; k += 8 )
                {
                    stage_sum += icvEvalHidHaarClassifierAVX(
                        cascade->stage_classifier[i].classifier + k,
                        variance_norm_factor, p_offset );
                }
            }
#endif
            for(; k < cascade->stage_classifier[i].count; k++ )
            {

                stage_sum += icvEvalHidHaarClassifier(
                    cascade->stage_classifier[i].classifier + k,
                    variance_norm_factor, p_offset );
            }

            if( stage_sum < cascade->stage_classifier[i].threshold )
                return -i;
        }
    }
    return 1;
}


double icvEvalHidHaarClassifier( CvHidHaarClassifier* classifier,
                                 double variance_norm_factor,
                                 size_t p_offset )
{
    int idx = 0;
    /*#if CV_HAAR_USE_SSE && !CV_HAAR_USE_AVX
        if(cv::checkHardwareSupport(CV_CPU_SSE2))//based on old SSE variant. Works slow
        {
            double CV_DECL_ALIGNED(16) temp[2];
            __m128d zero = _mm_setzero_pd();
            do
            {
                CvHidHaarTreeNode* node = classifier->node + idx;
                __m128d t = _mm_set1_pd((node->threshold)*variance_norm_factor);
                __m128d left = _mm_set1_pd(node->left);
                __m128d right = _mm_set1_pd(node->right);

                double _sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
                _sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
                if( node->feature.rect[2].p0 )
                    _sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;

                __m128d sum = _mm_set1_pd(_sum);
                t = _mm_cmplt_sd(sum, t);
                sum = _mm_blendv_pd(right, left, t);

                _mm_store_pd(temp, sum);
                idx = (int)temp[0];
            }
            while(idx > 0 );

        }
        else
    #endif*/
    {
        do
        {
            CvHidHaarTreeNode* node = classifier->node + idx;
            double t = node->threshold * variance_norm_factor;

            double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
            sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;

            if( node->feature.rect[2].p0 )
                sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;

            idx = sum < t ? node->left : node->right;
        }
        while( idx > 0 );
    }
    return classifier->alpha[-idx];
}

CvSeq*
cvHaarDetectObjectsMultiCameraForROC( const CvArr* _img,
                     CvHaarClassifierCascade* cascade, CvMemStorage* storage,
                     std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                     double scaleFactor, int minNeighbors, int flags,
                     CvSize minSize, CvSize maxSize, bool outputRejectLevels )
{
    const double GROUP_EPS = 0.2;
    CvMat stub, *img = (CvMat*)_img;
    cv::Ptr<CvMat> temp, sum, tilted, sqsum, normImg, sumcanny, imgSmall;
    CvSeq* result_seq = 0;
    cv::Ptr<CvMemStorage> temp_storage;

    std::vector<cv::Rect> allCandidates;
    std::vector<cv::Rect> rectList;
    std::vector<int> rweights;
    double factor;
    int coi;
    bool doCannyPruning = (flags & CV_HAAR_DO_CANNY_PRUNING) != 0;
    bool findBiggestObject = (flags & CV_HAAR_FIND_BIGGEST_OBJECT) != 0;
    bool roughSearch = (flags & CV_HAAR_DO_ROUGH_SEARCH) != 0;
    cv::Mutex mtx;

    if( !CV_IS_HAAR_CLASSIFIER(cascade) )
        CV_Error( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier cascade" );

    if( !storage )
        CV_Error( CV_StsNullPtr, "Null storage pointer" );

    img = cvGetMat( img, &stub, &coi );
    if( coi )
        CV_Error( CV_BadCOI, "COI is not supported" );

    if( CV_MAT_DEPTH(img->type) != CV_8U )
        CV_Error( CV_StsUnsupportedFormat, "Only 8-bit images are supported" );

    if( scaleFactor <= 1 )
        CV_Error( CV_StsOutOfRange, "scale factor must be > 1" );

    if( findBiggestObject )
        flags &= ~CV_HAAR_SCALE_IMAGE;

    if( maxSize.height == 0 || maxSize.width == 0 )
    {
        maxSize.height = img->rows;
        maxSize.width = img->cols;
    }

    temp = cvCreateMat( img->rows, img->cols, CV_8UC1 );
    sum = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 );
    sqsum = cvCreateMat( img->rows + 1, img->cols + 1, CV_64FC1 );

    if( !cascade->hid_cascade )
        icvCreateHidHaarClassifierCascade(cascade);

    if( cascade->hid_cascade->has_tilted_features )
        tilted = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 );

    result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );

    if( CV_MAT_CN(img->type) > 1 )
    {
        cvCvtColor( img, temp, CV_BGR2GRAY );
        img = temp;
    }

    if( findBiggestObject )
        flags &= ~(CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING);

    if( flags & CV_HAAR_SCALE_IMAGE )
    {
        CvSize winSize0 = cascade->orig_window_size;
#ifdef HAVE_IPP
        int use_ipp = cascade->hid_cascade->ipp_stages != 0;

        if( use_ipp )
            normImg = cvCreateMat( img->rows, img->cols, CV_32FC1 );
#endif
        imgSmall = cvCreateMat( img->rows + 1, img->cols + 1, CV_8UC1 );

        for( factor = 1; ; factor *= scaleFactor )
        {
            CvSize winSize = { cvRound(winSize0.width*factor),
                                cvRound(winSize0.height*factor) };
            CvSize sz = { cvRound( img->cols/factor ), cvRound( img->rows/factor ) };
            CvSize sz1 = { sz.width - winSize0.width + 1, sz.height - winSize0.height + 1 };

            CvRect equRect = { icv_object_win_border, icv_object_win_border,
                winSize0.width - icv_object_win_border*2,
                winSize0.height - icv_object_win_border*2 };

            CvMat img1, sum1, sqsum1, norm1, tilted1, mask1;
            CvMat* _tilted = 0;

            if( sz1.width <= 0 || sz1.height <= 0 )
                break;
            if( winSize.width > maxSize.width || winSize.height > maxSize.height )
                break;
            if( winSize.width < minSize.width || winSize.height < minSize.height )
                continue;

            img1 = cvMat( sz.height, sz.width, CV_8UC1, imgSmall->data.ptr );
            sum1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, sum->data.ptr );
            sqsum1 = cvMat( sz.height+1, sz.width+1, CV_64FC1, sqsum->data.ptr );
            if( tilted )
            {
                tilted1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, tilted->data.ptr );
                _tilted = &tilted1;
            }
            norm1 = cvMat( sz1.height, sz1.width, CV_32FC1, normImg ? normImg->data.ptr : 0 );
            mask1 = cvMat( sz1.height, sz1.width, CV_8UC1, temp->data.ptr );

            cvResize( img, &img1, CV_INTER_LINEAR );
            cvIntegral( &img1, &sum1, &sqsum1, _tilted );

            int ystep = factor > 2 ? 1 : 2;
            const int LOCS_PER_THREAD = 1000;
            int stripCount = ((sz1.width/ystep)*(sz1.height + ystep-1)/ystep + LOCS_PER_THREAD/2)/LOCS_PER_THREAD;
            stripCount = std::min(std::max(stripCount, 1), 100);

#ifdef HAVE_IPP
            if( use_ipp )
            {
                cv::Mat fsum(sum1.rows, sum1.cols, CV_32F, sum1.data.ptr, sum1.step);
                cv::Mat(&sum1).convertTo(fsum, CV_32F, 1, -(1<<24));
            }
            else
#endif
                cvSetImagesForHaarClassifierCascade( cascade, &sum1, &sqsum1, _tilted, 1. );

            cv::Mat _norm1(&norm1), _mask1(&mask1);
            cv::parallel_for_(cv::Range(0, stripCount),
                         cv::HaarDetectObjects_ScaleImage_Invoker(cascade,
                                (((sz1.height + stripCount - 1)/stripCount + ystep-1)/ystep)*ystep,
                                factor, cv::Mat(&sum1), cv::Mat(&sqsum1), &_norm1, &_mask1,
                                cv::Rect(equRect), allCandidates, rejectLevels, levelWeights, outputRejectLevels, &mtx));
        }
    }
    else
    {
        int n_factors = 0;
        cv::Rect scanROI;

        cvIntegralMultiImage( img, sum, sqsum, tilted );

        if( doCannyPruning )
        {
            sumcanny = cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 );
            cvCanny( img, temp, 0, 50, 3 );
            cvIntegral( temp, sumcanny );
        }

        for( n_factors = 0, factor = 1;
             factor*cascade->orig_window_size.width < img->cols - 10 &&
             factor*cascade->orig_window_size.height < img->rows - 10;
             n_factors++, factor *= scaleFactor )
            ;

        if( findBiggestObject )
        {
            scaleFactor = 1./scaleFactor;
            factor *= scaleFactor;
        }
        else
            factor = 1;

        for( ; n_factors-- > 0; factor *= scaleFactor )
        {
            const double ystep = std::max( 2., factor );
            CvSize winSize = { cvRound( cascade->orig_window_size.width * factor ),
                                cvRound( cascade->orig_window_size.height * factor )};
            CvRect equRect = { 0, 0, 0, 0 };
            int *p[4] = {0,0,0,0};
            int *pq[4] = {0,0,0,0};
            int startX = 0, startY = 0;
            int endX = cvRound((img->cols - winSize.width) / ystep);
            int endY = cvRound((img->rows - winSize.height) / ystep);

            if( winSize.width < minSize.width || winSize.height < minSize.height )
            {
                if( findBiggestObject )
                    break;
                continue;
            }

            if ( winSize.width > maxSize.width || winSize.height > maxSize.height )
            {
                if( !findBiggestObject )
                    break;
                continue;
            }

            cvSetImagesForHaarClassifierCascade( cascade, sum, sqsum, tilted, factor );
            cvZero( temp );

            if( doCannyPruning )
            {
                equRect.x = cvRound(winSize.width*0.15);
                equRect.y = cvRound(winSize.height*0.15);
                equRect.width = cvRound(winSize.width*0.7);
                equRect.height = cvRound(winSize.height*0.7);

                p[0] = (int*)(sumcanny->data.ptr + equRect.y*sumcanny->step) + equRect.x;
                p[1] = (int*)(sumcanny->data.ptr + equRect.y*sumcanny->step)
                            + equRect.x + equRect.width;
                p[2] = (int*)(sumcanny->data.ptr + (equRect.y + equRect.height)*sumcanny->step) + equRect.x;
                p[3] = (int*)(sumcanny->data.ptr + (equRect.y + equRect.height)*sumcanny->step)
                            + equRect.x + equRect.width;

                pq[0] = (int*)(sum->data.ptr + equRect.y*sum->step) + equRect.x;
                pq[1] = (int*)(sum->data.ptr + equRect.y*sum->step)
                            + equRect.x + equRect.width;
                pq[2] = (int*)(sum->data.ptr + (equRect.y + equRect.height)*sum->step) + equRect.x;
                pq[3] = (int*)(sum->data.ptr + (equRect.y + equRect.height)*sum->step)
                            + equRect.x + equRect.width;
            }

            if( scanROI.area() > 0 )
            {
                //adjust start_height and stop_height
                startY = cvRound(scanROI.y / ystep);
                endY = cvRound((scanROI.y + scanROI.height - winSize.height) / ystep);

                startX = cvRound(scanROI.x / ystep);
                endX = cvRound((scanROI.x + scanROI.width - winSize.width) / ystep);
            }

            cv::parallel_for_(cv::Range(startY, endY),
                cv::HaarDetectObjects_ScaleCascade_Invoker(cascade, winSize, cv::Range(startX, endX),
                                                           ystep, sum->step, (const int**)p,
                                                           (const int**)pq, allCandidates, &mtx ));

            if( findBiggestObject && !allCandidates.empty() && scanROI.area() == 0 )
            {
                rectList.resize(allCandidates.size());
                std::copy(allCandidates.begin(), allCandidates.end(), rectList.begin());

                groupRectangles(rectList, std::max(minNeighbors, 1), GROUP_EPS);

                if( !rectList.empty() )
                {
                    size_t i, sz = rectList.size();
                    cv::Rect maxRect;

                    for( i = 0; i < sz; i++ )
                    {
                        if( rectList[i].area() > maxRect.area() )
                            maxRect = rectList[i];
                    }

                    allCandidates.push_back(maxRect);

                    scanROI = maxRect;
                    int dx = cvRound(maxRect.width*GROUP_EPS);
                    int dy = cvRound(maxRect.height*GROUP_EPS);
                    scanROI.x = std::max(scanROI.x - dx, 0);
                    scanROI.y = std::max(scanROI.y - dy, 0);
                    scanROI.width = std::min(scanROI.width + dx*2, img->cols-1-scanROI.x);
                    scanROI.height = std::min(scanROI.height + dy*2, img->rows-1-scanROI.y);

                    double minScale = roughSearch ? 0.6 : 0.4;
                    minSize.width = cvRound(maxRect.width*minScale);
                    minSize.height = cvRound(maxRect.height*minScale);
                }
            }
        }
    }

    rectList.resize(allCandidates.size());
    if(!allCandidates.empty())
        std::copy(allCandidates.begin(), allCandidates.end(), rectList.begin());

    if( minNeighbors != 0 || findBiggestObject )
    {
        if( outputRejectLevels )
        {
            groupRectangles(rectList, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
        }
        else
        {
            groupRectangles(rectList, rweights, std::max(minNeighbors, 1), GROUP_EPS);
        }
    }
    else
        rweights.resize(rectList.size(),0);

    if( findBiggestObject && rectList.size() )
    {
        CvAvgComp result_comp = {{0,0,0,0},0};

        for( size_t i = 0; i < rectList.size(); i++ )
        {
            cv::Rect r = rectList[i];
            if( r.area() > cv::Rect(result_comp.rect).area() )
            {
                result_comp.rect = r;
                result_comp.neighbors = rweights[i];
            }
        }
        cvSeqPush( result_seq, &result_comp );
    }
    else
    {
        for( size_t i = 0; i < rectList.size(); i++ )
        {
            CvAvgComp c;
            c.rect = rectList[i];
            c.neighbors = !rweights.empty() ? rweights[i] : 0;
            cvSeqPush( result_seq, &c );
        }
    }

    return result_seq;
}


void
cvIntegralMultiImage( const CvArr* image, CvArr* sumImage,
            CvArr* sumSqImage, CvArr* tiltedSumImage )
{
    cv::Mat src = cv::cvarrToMat(image), sum = cv::cvarrToMat(sumImage), sum0 = sum;
    cv::Mat sqsum0, sqsum, tilted0, tilted;
    cv::Mat *psqsum = 0, *ptilted = 0;

    if( sumSqImage )
    {
        sqsum0 = sqsum = cv::cvarrToMat(sumSqImage);
        psqsum = &sqsum;
    }

    if( tiltedSumImage )
    {
        tilted0 = tilted = cv::cvarrToMat(tiltedSumImage);
        ptilted = &tilted;
    }
    cv::integralMultiImage( src, sum, psqsum ? cv::_OutputArray(*psqsum) : cv::_OutputArray(),
                  ptilted ? cv::_OutputArray(*ptilted) : cv::_OutputArray(), sum.depth() );

    CV_Assert( sum.data == sum0.data && sqsum.data == sqsum0.data && tilted.data == tilted0.data );
}