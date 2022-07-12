package com.example.blind;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Switch;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraGLSurfaceView;
import org.opencv.android.Utils;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.LoaderCallbackInterface;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.imgproc.Imgproc.cvtColor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.List;

public class CameraOpencv extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    private static final String MODEL_PATH= "midasModel.tflite";

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException{
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel1 = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel1.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private int inputImageDim = 256;
    private float mean[] = {123.675f ,  116.28f ,  103.53f };
    private float std[] = { 58.395f , 57.12f ,  57.375f };
    private long oldTime = Calendar.getInstance().getTimeInMillis();
    private float max = 0;
    private float min = 0;
    private int cameraMode = 0;

    private Interpreter tflite;

    private ImageProcessor inputTensorProcessor = new ImageProcessor.Builder()
            .add( new ResizeOp( inputImageDim , inputImageDim , ResizeOp.ResizeMethod.BILINEAR ) )
            .add( new NormalizeOp( mean , std ) )
            .build();

    private Mat mRGBD;
    private CameraBridgeViewBase cameraBridgeViewBase;
    private BaseLoaderCallback baseLoaderCallback =  new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:{
                    Log.i(TAG, "onManagerConnected: Opencv loaded");
                    cameraBridgeViewBase.enableView();
                }
                default:{
                    super.onManagerConnected(status);
                }
                break;
            }
            super.onManagerConnected(status);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ActivityCompat.requestPermissions(CameraOpencv.this, new String[]{Manifest.permission.CAMERA}, 1);
        setContentView(R.layout.activity_camera_opencv);
        cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.camera_surface);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        Button button = (Button) findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if(cameraMode == 0)
                    cameraMode = 1;
                else
                    cameraMode = 0;
            }
        });

        try {
            initTF();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void initTF() throws IOException {
        tflite = new Interpreter(loadModelFile(this));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode){
            case 1:{
                if(grantResults.length >  0 &&  grantResults[0] == PackageManager.PERMISSION_GRANTED){
                    cameraBridgeViewBase.setCameraPermissionGranted();
                }else{

                }
                return;
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(OpenCVLoader.initDebug()){
            Log.d(TAG, "onResume: Opencv initialized");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }else{
            Log.d(TAG, "onResume: Opencv not initialized");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase != null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRGBD = makeDeepImage(inputFrame.rgba());
        setFrameCount();
        return mRGBD;
    }

    private void setFrameCount(){
        long currentTime = Calendar.getInstance().getTimeInMillis();
        TextView helpText = (TextView) findViewById(R.id.outputtext);
        helpText.setText("FPS : " + String.valueOf((float)1000/(currentTime - oldTime)));
        oldTime = currentTime;
    }

    public Mat makeDeepImage(Mat mRGBA){

        TensorImage inputTensor = TensorImage.fromBitmap(converMat2Bitmat(mRGBA));
        inputTensor = inputTensorProcessor.process( inputTensor );
        TensorBuffer outputTensor = TensorBufferFloat.createFixedSize(new int[]{inputImageDim, inputImageDim, 1}, DataType.FLOAT32 );

        tflite.run(inputTensor.getBuffer(), outputTensor.getBuffer());

        outputTensor = scaling(outputTensor);
        Bitmap outmap = byteBufferToBitmap( outputTensor.getIntArray(), inputImageDim );
        checkForObjects(outmap);

        if(cameraMode == 0)
            return mRGBA;

        Mat output = new Mat(inputImageDim, inputImageDim, CvType.CV_8UC3);
        Utils.bitmapToMat(outmap, output);
        Size sz = new Size(mRGBA.width(), mRGBA.height());
        Imgproc.resize(output, output, sz);
        return output;
    }

    public Bitmap byteBufferToBitmap( int imageArray[], int imageDim){
        int pixels[] = imageArray;
        Bitmap bitmap = Bitmap.createBitmap(imageDim, imageDim, Bitmap.Config.RGB_565 );
        for ( int i = 0; i < imageDim; i++ ) {
            for ( int j = 0; j < imageDim; j ++) {
                int p = pixels[ i * imageDim + j ];
                bitmap.setPixel( j , i , Color.rgb( p , p , p ));
            }
        }
        return bitmap;
    }
    public Bitmap converMat2Bitmat (Mat img) {
        int width = img.width();
        int hight = img.height();
        Bitmap bmp;
        bmp = Bitmap.createBitmap(width, hight, Bitmap.Config.ARGB_8888);
        Mat tmp;
        tmp = img.channels()==1? new Mat(width, hight, CV_8UC1, new Scalar(1)): new Mat(width, hight, CvType.CV_8UC3, new Scalar(3));
        if (img.channels()==3)
            cvtColor(img, tmp, Imgproc.COLOR_RGB2BGRA);
        else if (img.channels()==1)
            cvtColor(img, tmp, Imgproc.COLOR_GRAY2RGBA);
        Utils.matToBitmap(img, bmp);
        return bmp;
    }

    private void checkForObjects(Bitmap img){

        long left = 0;
        long mid = 0;
        long right = 0;
        int biggest;

        for (int i = 0; i < img.getHeight(); i++) {
            for (int j = 0; j < img.getWidth(); j++) {

                if(j < (int)(img.getWidth() / 3) && (Math.abs(img.getPixel(i, j)) / 1000000) < MainActivity.threshold)
                    right += 1;

                if(j > (int)(img.getWidth() / 3) && j < (int)(img.getWidth() / 3) * 2 && (Math.abs(img.getPixel(i, j)) / 1000000) < MainActivity.threshold)
                    mid += 1;

                if(j > (int)(img.getWidth() / 3) * 2 && (Math.abs(img.getPixel(i, j)) / 1000000) < MainActivity.threshold)
                    left += 1;
            }
        }
        biggest = 4;
        if (left >= right && left >= mid && left > MainActivity.threshold2)
            biggest = 2;
        else if (right >= left && right >= mid && right> MainActivity.threshold2)
            biggest = 3;
        else if (mid >= left && mid >= right &&  mid > MainActivity.threshold2)
            biggest = 1;

        makeSound(biggest);
    }

    private void makeSound(int number){
        MediaPlayer mp;
        switch(number){
            case 1:
                mp = MediaPlayer.create(getApplicationContext(), R.raw.beeps);
                mp.start();
                break;
            case 2:
                mp = MediaPlayer.create(getApplicationContext(), R.raw.leftbeeps);
                mp.start();
                break;
            case 3:
                mp = MediaPlayer.create(getApplicationContext(), R.raw.rightbeeps);
                mp.start();
                break;
            case 4:
                break;
        }
    }

    public TensorBuffer scaling(TensorBuffer input) {

        float pixel[]  = input.getFloatArray();

        max = pixel[0];
        min = pixel[0];
        for (int i = 1; i < pixel.length; i++) {
            if (pixel[i] > max) {
                max = pixel[i];
            }
            if (pixel[i] < min) {
                min = pixel[i];
            }
        }
        float diffWithScaleFactor = 255 / (max - min) ;
        float p;
        for (int i = 0; i < pixel.length; i++) {

            p = ( pixel[ i ] - min ) * diffWithScaleFactor;
            if ( p < 0 ) {
                p += 255;
            }
            pixel[ i ] = p;
        }
        TensorBuffer output = TensorBufferFloat.createFixedSize(input.getShape() , DataType.FLOAT32 );
        output.loadArray( pixel );
        return output;
        }

}