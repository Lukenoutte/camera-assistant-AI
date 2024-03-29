/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Handler;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Timer;
import java.util.TimerTask;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;
  private TextToSpeech t1;
  private Classifier detector;
  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private Bitmap cropCopyBitmap2 = null;
  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {

     t1 = new TextToSpeech(getApplicationContext(), status -> {
          if (status != TextToSpeech.ERROR) {
              t1.setLanguage(Locale.ENGLISH);
          }
      });

    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            canvas -> {
              tracker.draw(canvas);

              if (isDebug()) {
                tracker.drawDebug(canvas);
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {


    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
            () -> {
                LOGGER.i("Running detection on image " + currTimestamp);
                final long startTime = SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                cropCopyBitmap2 = Bitmap.createBitmap(croppedBitmap);
                final Canvas canvas1 = new Canvas(cropCopyBitmap);
                final Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Style.STROKE);
                paint.setStrokeWidth(2.0f);

                float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                switch (MODE) {
                    case TF_OD_API:
                        minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        break;
                }

                // Lista de objetos
                List<Classifier.Recognition> mappedRecognitions =
                        new LinkedList<Classifier.Recognition>();



                for (final Classifier.Recognition result : results) {
                    final RectF location = result.getLocation();
                    if (location != null && result.getConfidence() >= minimumConfidence) {
                        canvas1.drawRect(location, paint);

                        cropToFrameTransform.mapRect(location);

                        result.setLocation(location);
                        mappedRecognitions.add(result);
                    }
                }
                  // Fala os objetos encontrados

                if (mappedRecognitions.isEmpty() == false){
                    String toSpeak = mappedRecognitions.get(0).getTitle();
                    int divScreenWeight = previewHeight/3;
                    int divScreenHeight =  previewWidth/3;
                    int div2ScreenWeight = divScreenWeight * 2;
                    int div2ScreenHeight= divScreenHeight * 2;
                    int metadeDivScreenWeight = divScreenWeight/2;
                    int metadeDivScreenHeight = divScreenHeight/2;
                    float  topRetangle = mappedRecognitions.get(0).getLocation().left;
                    float  rightRetangle = mappedRecognitions.get(0).getLocation().top;
                    float  bottomRetangle = mappedRecognitions.get(0).getLocation().right;
                    float  leftRetangle = mappedRecognitions.get(0).getLocation().bottom;
                    boolean onCenter = false;
                    boolean onTop = false;
                    boolean onBottom = false;
                    boolean onLeft = false;
                    boolean onRight = false;
                    boolean onFullH = false;
                    boolean onFullV = false;
                    String position = " ";


                    // Check if retangle is on right
                    if (rightRetangle < divScreenWeight){
                        position += " on the right";
                        onRight = true;
                    }

                    if(onRight == true && leftRetangle > (divScreenWeight + metadeDivScreenWeight)){
                        position += " and in the center";
                    }

                    if(onRight == true && bottomRetangle > (div2ScreenHeight + metadeDivScreenHeight)){
                        position += " and on the bottom";
                    }
                    if(onRight == true && topRetangle < (divScreenHeight - metadeDivScreenHeight)){
                        position += " and on the top";
                    }
                    // finish on right


                    // Check if retangle is on center
                    if (rightRetangle > (divScreenWeight - metadeDivScreenWeight)){
                        position = " in the center";
                        onCenter = true;
                    }
                    if(onCenter == true && leftRetangle > (div2ScreenWeight + metadeDivScreenWeight)){
                        position += " and on the left";
                    }
                    if(onCenter == true && bottomRetangle > (div2ScreenHeight + metadeDivScreenHeight)){
                        position += " and on the bottom";
                    }
                    if(onCenter == true && topRetangle < (divScreenHeight - metadeDivScreenHeight)){
                        position += " and on the top";
                    }
                    // finish on center

                    // Check if retangle is on left
                    if (rightRetangle > (divScreenWeight + metadeDivScreenWeight)){
                        position = " on the left";
                        onLeft = true;
                    }
                    if(onLeft == true && bottomRetangle > (div2ScreenHeight + metadeDivScreenHeight)){
                        position += " and on the bottom";
                    }
                    if(onLeft == true && topRetangle < (divScreenHeight - metadeDivScreenHeight)){
                        position += " and on the top";
                    }
                    // finish on left

                    // Full screen
                    // Horizontal
                    if(rightRetangle < (divScreenWeight - metadeDivScreenWeight) && leftRetangle > (div2ScreenWeight + metadeDivScreenWeight)){
                        position = "  ";
                        onFullH = true;
                    }
                    if(onFullH == true && topRetangle < (divScreenHeight - metadeDivScreenHeight)){
                        position += " on the top";
                    }
                    if(onFullH == true && bottomRetangle > (div2ScreenHeight + metadeDivScreenHeight)){
                        position += " on the bottom";
                    }
                    if(onFullH == true && topRetangle > (divScreenHeight - metadeDivScreenHeight) && bottomRetangle < (div2ScreenHeight + metadeDivScreenHeight)){
                        position += " in the center";
                    }

                    //Vertical
                    if(topRetangle < (divScreenHeight - metadeDivScreenHeight) && bottomRetangle > (div2ScreenHeight + metadeDivScreenHeight)){
                        position = "  ";
                        onFullV = true;
                    }
                    if (onFullV == true && rightRetangle > (divScreenWeight + metadeDivScreenWeight)){
                        position = " on the left";
                    }
                    if (onFullV == true && rightRetangle < (divScreenWeight - metadeDivScreenWeight)){
                        position += " on the right";
                        }
                    if (onFullV == true && rightRetangle < (divScreenWeight + metadeDivScreenWeight) && rightRetangle > (divScreenWeight - metadeDivScreenWeight) ){
                        position = " on the center";
                    }

                    // full
                    if(rightRetangle < (divScreenWeight - metadeDivScreenWeight) && leftRetangle > (div2ScreenWeight + metadeDivScreenWeight) && topRetangle < (divScreenHeight - metadeDivScreenHeight) && bottomRetangle > (div2ScreenHeight + metadeDivScreenHeight)){
                        position = " on full screen";
                        }
                    // finish Full screen
                    toSpeak += position;
                    Toast.makeText(getApplicationContext(), toSpeak, Toast.LENGTH_SHORT);

                    t1.speak(toSpeak, TextToSpeech.QUEUE_FLUSH, null);

                  }else{
                    String toError = "Sorry, Try again!";
                    Toast.makeText(getApplicationContext(), toError, Toast.LENGTH_SHORT);

                    t1.speak(toError, TextToSpeech.QUEUE_FLUSH, null);
                }


                tracker.trackResults(mappedRecognitions, currTimestamp);
                trackingOverlay.postInvalidate();

              computingDetection = false;

              runOnUiThread(
                      () -> {
                        showFrameInfo(previewWidth + "x" + previewHeight);
                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                        showInference(lastProcessingTimeMs + "ms");
                      });
                final Handler handler = new Handler();
                handler.postDelayed(() -> {

                    Log.e("Hey", "CLEAR");

                    //delay 2s
                }, 2000);

            });

  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
