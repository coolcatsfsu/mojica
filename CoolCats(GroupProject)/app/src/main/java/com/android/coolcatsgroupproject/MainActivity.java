package com.android.coolcatsgroupproject;

import java.util.LinkedList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.concurrent.atomic.AtomicBoolean;

import android.graphics.Path;
import android.hardware.Camera;
import android.support.v7.app.AppCompatActivity;
import android.graphics.SumPathEffect;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.coolcatsgroupproject.colorIdentifier.IdentifyColor;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

public class MainActivity extends AppCompatActivity implements OnTouchListener, CvCameraViewListener2 {

    private static final int JAVA_DETECTOR =0, NATIVE_DETECTOR = 1;

    private static final String TAG = "MainActivity";
    static {
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV not loaded");
        }
        else {
            Log.d(TAG, "OpenCV loaded");
        }
    }

    private Mat rgba, gray, intermed, spectrum;
    private int detectorType = JAVA_DETECTOR;
    private CustomSurfaceView OpenCV_CameraView;
    private List<Size> resolutionList;
    private SeekBar seekBarMINthreshold = null, seekBarMAXthreshold = null;
    private TextView seekBarTextMINthreshold = null, fingerNumberTEXT = null;
    double thresholds = 0;
    private Scalar rgbaColor, hsvColor, OUTLINE_COLOR, OUTLINE_COLOR_WHITE;
    private IdentifyColor colorIdentifier;
    private boolean colorON = false;
    private Size SPECTRUM_SIZE;

    final Handler handler = new Handler();

    int fingerNumber = 0;
    final Runnable fingerNumberSYNC = new Runnable() {
        @Override
        public void run() {
            SyncFingerNumber();
        }
    };

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded");
                    OpenCV_CameraView.enableView();
                    OpenCV_CameraView.setOnTouchListener(MainActivity.this);
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "OnCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        OpenCV_CameraView = (CustomSurfaceView) findViewById(R.id.main_surface_view);
        if (OpenCV_CameraView != null) {
            OpenCV_CameraView.setCvCameraViewListener(this);
        }

        seekBarTextMINthreshold = (TextView) findViewById(R.id.textView3);
        fingerNumberTEXT = (TextView) findViewById(R.id.numberOfFingers);

        seekBarMINthreshold = (SeekBar) findViewById(R.id.seekBar1);
        seekBarMINthreshold.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            int progression = 0;

            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                progression = progress;
                seekBarTextMINthreshold.setText(String.valueOf(progression));
            }

            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            public void onStopTrackingTouch(SeekBar seekBar) {
                seekBarTextMINthreshold.setText(String.valueOf(progression));
            }
        });
        seekBarMINthreshold.setProgress(8700);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        intermed = new Mat();
        rgba = new Mat();
        gray = new Mat();

        Camera.Size resolution = OpenCV_CameraView.Resolution();
        String resolution_info = "Resolution " + Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
        Toast.makeText(this, resolution_info, Toast.LENGTH_SHORT).show();
        Camera.Parameters parameters = OpenCV_CameraView.Parameter();
        parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_INFINITY);
        Toast.makeText(this, "Focus mode: " + parameters.getFocusMode(), Toast.LENGTH_SHORT).show();

        rgba = new Mat(height, width, CvType.CV_8UC4);
        colorIdentifier = new IdentifyColor();
        spectrum = new Mat();
        rgbaColor = new Scalar(225);
        hsvColor = new Scalar(225);
        SPECTRUM_SIZE = new Size(200, 64);
        OUTLINE_COLOR = new Scalar(225, 0, 0, 225);
        OUTLINE_COLOR_WHITE = new Scalar(225, 225, 225, 225);
    }

    @Override
    public void onCameraViewStopped() {
        gray.release();
        rgba.release();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        gray = inputFrame.gray();
        rgba = inputFrame.rgba();
        thresholds = seekBarMINthreshold.getProgress();

        Imgproc.GaussianBlur(rgba, rgba, new Size(3, 3), 1, 1);

        if (!colorON) {
            return rgba;
        }

        List<MatOfPoint> outlines = colorIdentifier.OutlinesARE();
        colorIdentifier.Analyze(rgba);
        if (outlines.size() <= 0) {
            return rgba;
        }

        RotatedRect rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(outlines.get(0).toArray()));

        int n, limit;
        n = 0;
        limit = 0;
        double limit_height, limit_width;
        limit_height = rotatedRect.size.height;
        limit_width = rotatedRect.size.width;
        while (n < outlines.size()) {
            rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(outlines.get(n).toArray()));

            if (limit_height * limit_width < rotatedRect.size.height * rotatedRect.size.width) {
                limit_height = rotatedRect.size.height;
                limit_width = rotatedRect.size.width;
                limit = n;
            }
            n++;
        }

        Rect limitRect = Imgproc.boundingRect(new MatOfPoint(outlines.get(limit).toArray()));
        Imgproc.rectangle(rgba, limitRect.tl(), limitRect.br(), OUTLINE_COLOR_WHITE, 2, 8, 0);

        int height_thresholdRect;
        height_thresholdRect = 0;
        double m;
        m = limitRect.br().y - limitRect.tl().y;
        m = 0.7 * m;
        m = limitRect.tl().y + m;

        Imgproc.rectangle(rgba, limitRect.tl(), new Point(limitRect.br().x, m), OUTLINE_COLOR, 2, 8, 0);

        MatOfPoint2f matAmount;
        MatOfInt hull;
        MatOfInt4 convexus;
        matAmount = new MatOfPoint2f();
        hull = new MatOfInt();
        convexus = new MatOfInt4();
        Imgproc.approxPolyDP(new MatOfPoint2f(outlines.get(limit).toArray()), matAmount, 3, true);
        outlines.set(limit, new MatOfPoint(matAmount.toArray()));
        Imgproc.convexHull(new MatOfPoint(outlines.get(limit).toArray()), hull);

        if (3 > hull.toArray().length) {
            return rgba;
        }

        Imgproc.convexityDefects(new MatOfPoint(outlines.get(limit).toArray()), hull, convexus);

        List<MatOfPoint> hullAmount;
        List<Point> listAmount;
        hullAmount = new LinkedList<MatOfPoint>();
        listAmount = new LinkedList<Point>();
        int o;
        o = 0;
        while (o < hull.toList().size()) {
            listAmount.add(outlines.get(limit).toList().get(hull.toList().get(o)));
            o++;
        }

        MatOfPoint p;
        p = new MatOfPoint();
        p.fromList(listAmount);
        hullAmount.add(p);

        List<MatOfPoint> defectAmount;
        defectAmount = new LinkedList<MatOfPoint>();
        List<Point> listAmountDefect;
        listAmountDefect = new LinkedList<Point>();
        int q;
        q = 0;
        while (q < convexus.toList().size()) {
            Point farAmount = outlines.get(limit).toList().get(convexus.toList().get(q + 2));
            Integer depth = convexus.toList().get(q + 3);

            if (m > farAmount.y && thresholds < depth) {
                listAmountDefect.add(outlines.get(limit).toList().get(convexus.toList().get(q + 2)));
            }
            q = q+4;
        }

        MatOfPoint r;
        r = new MatOfPoint();
        r.fromList(listAmount);
        defectAmount.add(r);

        Imgproc.drawContours(rgba, hullAmount, -1, OUTLINE_COLOR, 3);

        int totalDefects;
        totalDefects = (int) convexus.total();

        this.fingerNumber = listAmountDefect.size();
        if (5 < this.fingerNumber) {
            this.fingerNumber = 5;
        }

        handler.post(fingerNumberSYNC);

        for (Point s : listAmountDefect) {
            Imgproc.circle(rgba, s, 6, new Scalar(255, 0, 255));
        }

        return rgba;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int rows, columns, x, y, x_offset, y_offset;
        rows = rgba.rows();
        columns = rgba.cols();
        y_offset = (OpenCV_CameraView.getWidth() - rows) / 2;
        x_offset = (OpenCV_CameraView.getWidth() - columns) / 2;
        y = (int) event.getY() - y_offset;
        x = (int) event.getX() - x_offset;

        if ((y < 0) || (x < 0) || (y > rows) || (x > columns)) {
            return false;
        }

        Rect touch = new Rect();
        touch.y = (y > 5) ? y - 5 : 0;
        touch.x = (x > 5) ? x - 5 : 0;
        touch.height = (y + 5 < rows) ? y + 5 - touch.y : rows - touch.y;
        touch.width = (x + 5 < columns) ? x + 5 - touch.x : columns - touch.x;

        Mat rgba_touch_region, hsv_touch_region;
        rgba_touch_region = rgba.submat(touch);
        hsv_touch_region = new Mat();

        Imgproc.cvtColor(rgba_touch_region, hsv_touch_region, Imgproc.COLOR_RGB2HSV_FULL);

        hsvColor = Core.sumElems(hsv_touch_region);
        int n, amount;
        n = 0;
        amount = touch.height * touch.width;
        while (n < hsvColor.val.length) {
            hsvColor.val[n] /= amount;
            n++;
        }

        rgbaColor = hsvTOrgba(hsvColor);
        colorIdentifier.hsvON(hsvColor);
        Imgproc.resize(colorIdentifier.SpectrumIS(), spectrum, SPECTRUM_SIZE);
        colorON = true;

        rgba_touch_region.release();
        hsv_touch_region.release();

        return false;
    }

    private Scalar hsvTOrgba(Scalar hsvColor) {
        Mat rgba_amount, hsv_amount;
        rgba_amount = new Mat();
        hsv_amount = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(hsv_amount, rgba_amount, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(rgba_amount.get(0, 0));
    }

    public void SyncFingerNumber() {
        fingerNumberTEXT.setText(String.valueOf(this.fingerNumber));
    }

    @Override
    public void onPause() {
        super.onPause();
        if (OpenCV_CameraView != null) {
            OpenCV_CameraView.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, baseLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        OpenCV_CameraView.disableView();
    }
}
