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

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.DialogFragment;
import android.app.Fragment;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;

import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.Message;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.text.TextUtils;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.LayoutInflater;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.tensorflow.lite.examples.detection.customview.AutoFitTextureView;
import org.tensorflow.lite.examples.detection.env.Logger;

@SuppressLint("ValidFragment")
public class CameraConnectionFragment extends Fragment {
  private static final Logger LOGGER = new Logger();

  /**
   * The camera preview size will be chosen to be the smallest frame by pixel size capable of
   * containing a DESIRED_SIZE x DESIRED_SIZE square.
   */
  private static final int MINIMUM_PREVIEW_SIZE = 320;

  /** Conversion from screen rotation to JPEG orientation. */
  private static final SparseIntArray ORIENTATIONS = new SparseIntArray();

  private static final String FRAGMENT_DIALOG = "dialog";

  static {
    ORIENTATIONS.append(Surface.ROTATION_0, 90);
    ORIENTATIONS.append(Surface.ROTATION_90, 0);
    ORIENTATIONS.append(Surface.ROTATION_180, 270);
    ORIENTATIONS.append(Surface.ROTATION_270, 180);
  }

  /** A {@link Semaphore} to prevent the app from exiting before closing the camera. */
  private final Semaphore cameraOpenCloseLock = new Semaphore(1);
  /** A {@link OnImageAvailableListener} to receive frames as they are available. */
  private final OnImageAvailableListener imageListener;
  /** The input size in pixels desired by TensorFlow (width and height of a square bitmap). */
  private final Size inputSize;
  /** The layout identifier to inflate for this Fragment. */
  private final int layout;
  private Integer iso;

  private final ConnectionCallback cameraConnectionCallback;

  /** ID of the current {@link CameraDevice}. */
  private String cameraId;
  /** An {@link AutoFitTextureView} for camera preview. */
  private AutoFitTextureView textureView;
  /** A {@link CameraCaptureSession } for camera preview. */
  private CameraCaptureSession captureSession;
  /** A reference to the opened {@link CameraDevice}. */
  private CameraDevice cameraDevice;
  /** The rotation in degrees of the camera sensor from the display. */
  private Integer sensorOrientation;
  /** The {@link Size} of camera preview. */
  private Size previewSize;
  /** An additional thread for running tasks that shouldn't block the UI. */
  private HandlerThread backgroundThread;
  /** A {@link Handler} for running tasks in the background. */
  private Handler backgroundHandler;
  /** An {@link ImageReader} that handles preview frame capture. */

  /** {@link CaptureRequest.Builder} for the camera preview */
  private CaptureRequest.Builder previewRequestBuilder;
  /** {@link CaptureRequest} generated by {@link #previewRequestBuilder} */
  private CaptureRequest previewRequest;
  private TextToSpeech t2;
  /**
   * Camera state: Showing camera preview.
   */
  private static final int STATE_CLOSED = 0;

  /**
   * Camera state: Device is opened, but is not capturing.
   */
  private static final int STATE_OPENED = 1;

  /**
   * Camera state: Showing camera preview.
   */
  private static final int STATE_PREVIEW = 2;
  private boolean isoTest = false;
  /**
   * Camera state: Waiting for 3A convergence before capturing a photo.
   */
  private static final int STATE_WAITING_FOR_3A_CONVERGENCE = 3;




  /** {@link CameraDevice.StateCallback} is called when {@link CameraDevice} changes its state. */
  private final CameraDevice.StateCallback stateCallback =
          new CameraDevice.StateCallback() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onOpened(final CameraDevice cd) {
              // This method is called when the camera is opened.  We start camera preview here.
              cameraOpenCloseLock.release();
              cameraDevice = cd;
              createCameraPreviewSession();
            }

            @Override
            public void onDisconnected(final CameraDevice cd) {
              cameraOpenCloseLock.release();
              cd.close();
              cameraDevice = null;
            }

            @Override
            public void onError(final CameraDevice cd, final int error) {
              cameraOpenCloseLock.release();
              cd.close();
              cameraDevice = null;
              final Activity activity = getActivity();
              if (null != activity) {
                activity.finish();
              }
            }
          };

  private CameraConnectionFragment(
          final ConnectionCallback connectionCallback,
          final OnImageAvailableListener imageListener,
          final int layout,
          final Size inputSize) {
    this.cameraConnectionCallback = connectionCallback;
    this.imageListener = imageListener;
    this.layout = layout;
    this.inputSize = inputSize;
  }


  /**
   * Given {@code choices} of {@code Size}s supported by a camera, chooses the smallest one whose
   * width and height are at least as large as the minimum of both, or an exact match if possible.
   *
   * @param choices The list of sizes that the camera supports for the intended output class
   * @param width The minimum desired width
   * @param height The minimum desired height
   * @return The optimal {@code Size}, or an arbitrary one if none were big enough
   */
  protected static Size chooseOptimalSize(final Size[] choices, final int width, final int height) {
    final int minSize = Math.max(Math.min(width, height), MINIMUM_PREVIEW_SIZE);
    final Size desiredSize = new Size(width, height);

    // Collect the supported resolutions that are at least as big as the preview Surface
    boolean exactSizeFound = false;
    final List<Size> bigEnough = new ArrayList<Size>();
    final List<Size> tooSmall = new ArrayList<Size>();
    for (final Size option : choices) {
      if (option.equals(desiredSize)) {
        // Set the size but don't return yet so that remaining sizes will still be logged.
        exactSizeFound = true;
      }

      if (option.getHeight() >= minSize && option.getWidth() >= minSize) {
        bigEnough.add(option);
      } else {
        tooSmall.add(option);
      }
    }

    LOGGER.i("Desired size: " + desiredSize + ", min size: " + minSize + "x" + minSize);
    LOGGER.i("Valid preview sizes: [" + TextUtils.join(", ", bigEnough) + "]");
    LOGGER.i("Rejected preview sizes: [" + TextUtils.join(", ", tooSmall) + "]");

    if (exactSizeFound) {
      LOGGER.i("Exact size match found.");
      return desiredSize;
    }

    // Pick the smallest of those, assuming we found any
    if (bigEnough.size() > 0) {
      final Size chosenSize = Collections.min(bigEnough, new CompareSizesByArea());
      LOGGER.i("Chosen size: " + chosenSize.getWidth() + "x" + chosenSize.getHeight());
      return chosenSize;
    } else {
      LOGGER.e("Couldn't find any suitable preview size");
      return choices[0];
    }
  }

  public static CameraConnectionFragment newInstance(
          final ConnectionCallback callback,
          final OnImageAvailableListener imageListener,
          final int layout,
          final Size inputSize) {
    return new CameraConnectionFragment(callback, imageListener, layout, inputSize);
  }

  public static CameraConnectionFragment newInstance() {
  return null;
  }
  /**
   * Shows a {@link Toast} on the UI thread.
   *
   * @param text The message to show
   */
  private void showToast(final String text) {
    final Activity activity = getActivity();
    if (activity != null) {
      activity.runOnUiThread(
              new Runnable() {
                @Override
                public void run() {
                  Toast.makeText(activity, text, Toast.LENGTH_SHORT).show();
                }
              });
    }
  }

  @Override
  public View onCreateView(
          final LayoutInflater inflater, final ViewGroup container, final Bundle savedInstanceState) {
    return inflater.inflate(layout, container, false);
  }


  @Override
  public void onActivityCreated(final Bundle savedInstanceState) {
    super.onActivityCreated(savedInstanceState);


  }







  public void setCamera(String cameraId) {
    this.cameraId = cameraId;
  }




  /** Closes the current {@link CameraDevice}. */
  private void closeCamera() {
    mPendingUserCaptures = 0;
    mState = STATE_CLOSED;
    try {
      cameraOpenCloseLock.acquire();
      if (null != captureSession) {
        captureSession.close();
        captureSession = null;
      }
      if (null != cameraDevice) {
        cameraDevice.close();
        cameraDevice = null;
      }
      if (null != jpegImageReader) {
        jpegImageReader.close();
        jpegImageReader = null;
      }
    } catch (final InterruptedException e) {
      throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
    } finally {
      cameraOpenCloseLock.release();
    }
  }

  /** Starts a background thread and its {@link Handler}. */
  private void startBackgroundThread() {
    backgroundThread = new HandlerThread("ThreadCameraMan");
    backgroundThread.start();
    backgroundHandler = new Handler(backgroundThread.getLooper());

  }

  /** Stops the background thread and its {@link Handler}. */
  private void stopBackgroundThread() {
    backgroundThread.quitSafely();
    try {
      backgroundThread.join();
      backgroundThread = null;
      backgroundHandler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }
  }

  private final ImageReader.OnImageAvailableListener mOnJpegImageAvailableListener
          = new ImageReader.OnImageAvailableListener() {

    @Override
    public void onImageAvailable(ImageReader reader) {
      dequeueAndSaveImage(mJpegResultQueue, jpegImageReader);
    }

  };

  /** Creates a new {@link CameraCaptureSession} for camera preview. */

  private void createCameraPreviewSession() {
    final SurfaceTexture texture = textureView.getSurfaceTexture();


    // We configure the size of default buffer to be the size of camera preview we want.
    texture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());

    // This is the output Surface we need to start preview
    Surface surface = new Surface(texture);


    try {
      // We set up a CaptureRequest.Builder with the output Surface.


      previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);

      previewRequestBuilder.addTarget(surface);


      jpegImageReader.get().setOnImageAvailableListener(mOnJpegImageAvailableListener, backgroundHandler);

      // previewRequestBuilder.addTarget(jpegImageReader.get().getSurface());

      // Here, we create a CameraCaptureSession for camera preview.
      cameraDevice.createCaptureSession(
              Arrays.asList(surface, jpegImageReader.get().getSurface()),
              new CameraCaptureSession.StateCallback() {

                @Override
                public void onConfigured(final CameraCaptureSession cameraCaptureSession) {
                  // The camera is already closed
                  if (null == cameraDevice) {
                    return;
                  }

                  // When the session is ready, we start displaying the preview.
                  captureSession = cameraCaptureSession;
                  try {
                    // Auto focus should be continuous for camera preview.
                    previewRequestBuilder.set(
                            CaptureRequest.CONTROL_AF_MODE,
                            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);


                    // Finally, we start displaying the camera preview.
                    previewRequest = previewRequestBuilder.build();
                    captureSession.setRepeatingRequest(
                            previewRequest, mPreCaptureCallback, backgroundHandler);
                    mState = STATE_PREVIEW;
                  } catch (final CameraAccessException e) {
                    LOGGER.e(e, "Exception!");
                  }
                }

                @Override
                public void onConfigureFailed(final CameraCaptureSession cameraCaptureSession) {
                  showToast("Failed");
                }
              },
              backgroundHandler);

    } catch (CameraAccessException e) {
      e.printStackTrace();


    }


  }







  @RequiresApi(api = Build.VERSION_CODES.M)
  private void configureTransform(final int viewWidth, final int viewHeight) {
    final Activity activity = getActivity();
    if (null == textureView || null == previewSize || null == activity) {
      return;
    }
    final int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
    final Matrix matrix = new Matrix();
    final RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
    final RectF bufferRect = new RectF(0, 0, previewSize.getHeight(), previewSize.getWidth());
    final float centerX = viewRect.centerX();
    final float centerY = viewRect.centerY();
    if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
      bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
      matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
      final float scale =
              Math.max(
                      (float) viewHeight / previewSize.getHeight(),
                      (float) viewWidth / previewSize.getWidth());
      matrix.postScale(scale, scale, centerX, centerY);
      matrix.postRotate(90 * (rotation - 2), centerX, centerY);
    } else if (Surface.ROTATION_180 == rotation) {
      matrix.postRotate(180, centerX, centerY);
    }
    textureView.setTransform(matrix);

    // Start or restart the active capture session if the preview was initialized or
    // if its aspect ratio changed significantly.
    if (this.previewSize == null || !checkAspectsEqual(previewSize, this.previewSize)) {
      this.previewSize = previewSize;
      if (mState != STATE_CLOSED) createCameraPreviewSession();
    }
  }

  /**
   * Callback for Activities to use to initialize their data once the selected preview size is
   * known.
   */
  public interface ConnectionCallback {
    void onPreviewSizeChosen(Size size, int cameraRotation);
  }

  /** Compares two {@code Size}s based on their areas. */
  static class CompareSizesByArea implements Comparator<Size> {
    @Override
    public int compare(final Size lhs, final Size rhs) {
      // We cast here to ensure the multiplications won't overflow
      return Long.signum(
              (long) lhs.getWidth() * lhs.getHeight() - (long) rhs.getWidth() * rhs.getHeight());
    }
  }





  /** Shows an error message dialog. */
  public static class ErrorDialog extends DialogFragment {
    private static final String ARG_MESSAGE = "message";

    public static ErrorDialog newInstance(final String message) {
      final ErrorDialog dialog = new ErrorDialog();
      final Bundle args = new Bundle();
      args.putString(ARG_MESSAGE, message);
      dialog.setArguments(args);
      return dialog;
    }

    @Override
    public Dialog onCreateDialog(final Bundle savedInstanceState) {
      final Activity activity = getActivity();
      return new AlertDialog.Builder(activity)
              .setMessage(getArguments().getString(ARG_MESSAGE))
              .setPositiveButton(
                      android.R.string.ok,
                      new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(final DialogInterface dialogInterface, final int i) {
                          activity.finish();
                        }
                      })
              .setNegativeButton(android.R.string.cancel,
                      new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int which) {

                          if (activity != null) {
                            activity.finish();
                          }
                        }
                      })
              .create();
    }
  }

  // aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa





  static {
    ORIENTATIONS.append(Surface.ROTATION_0, 90);
    ORIENTATIONS.append(Surface.ROTATION_90, 0);
    ORIENTATIONS.append(Surface.ROTATION_180, 270);
    ORIENTATIONS.append(Surface.ROTATION_270, 180);
  }

  /**
   * Tag for the {@link Log}.
   */
  private static final String TAG = "CameraConnection";



  /**
   * Camera state: Waiting for the focus to be locked.
   */
  private static final int STATE_WAITING_LOCK = 1;

  /**
   * Camera state: Waiting for the exposure to be precapture state.
   */
  private static final int STATE_WAITING_PRECAPTURE = 2;

  /**
   * Camera state: Waiting for the exposure state to be something other than precapture.
   */
  private static final int STATE_WAITING_NON_PRECAPTURE = 3;

  /**
   * Camera state: Picture was taken.
   */
  private static final int STATE_PICTURE_TAKEN = 4;

  /**
   * Max preview width that is guaranteed by Camera2 API
   */
  private static final int MAX_PREVIEW_WIDTH = 1920;

  /**
   * Max preview height that is guaranteed by Camera2 API
   */
  private static final int MAX_PREVIEW_HEIGHT = 1080;




  /**
   * This is the output file for our picture.
   */
  private File mFile;










  /**
   * Sets up member variables related to camera.
   *
   * @param width  The width of available size for camera preview
   * @param height The height of available size for camera preview
   */
  /** Sets up member variables related to camera. */

  @RequiresApi(api = Build.VERSION_CODES.M)
  private void setUpCameraOutputs(int width, int height) {
    final Activity activity = getActivity();
    final CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
    try {
      final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);


      final StreamConfigurationMap map =
              characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);



      sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);

      // Danger, W.R.! Attempting to use too large a preview size could  exceed the camera
      // bus' bandwidth limitation, resulting in gorgeous previews but the storage of
      // garbage capture data.

      // For still image captures, we use the largest available size.
      Size largest = Collections.max(
              Arrays.asList(map.getOutputSizes(ImageFormat.YUV_420_888)),
              new CompareSizesByArea());


      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
        t2 = new TextToSpeech(getContext(), status -> {
          if (status != TextToSpeech.ERROR) {
            t2.setLanguage(Locale.ENGLISH);
          }
        });
      }


      // Find out if we need to swap dimension to get the preview size relative to sensor
      // coordinate.
      int displayRotation = activity.getWindowManager().getDefaultDisplay().getRotation();
      //noinspection ConstantConditions
      sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
      boolean swappedDimensions = false;
      switch (displayRotation) {
        case Surface.ROTATION_0:
        case Surface.ROTATION_180:
          if (sensorOrientation == 90 || sensorOrientation == 270) {
            swappedDimensions = true;
          }
          break;
        case Surface.ROTATION_90:
        case Surface.ROTATION_270:
          if (sensorOrientation == 0 || sensorOrientation == 180) {
            swappedDimensions = true;
          }
          break;
        default:
          Log.e(TAG, "Display rotation is invalid: " + displayRotation);
      }

      Point displaySize = new Point();
      activity.getWindowManager().getDefaultDisplay().getSize(displaySize);
      int rotatedPreviewWidth = width;
      int rotatedPreviewHeight = height;
      int maxPreviewWidth = displaySize.x;
      int maxPreviewHeight = displaySize.y;

      if (swappedDimensions) {
        rotatedPreviewWidth = height;
        rotatedPreviewHeight = width;
        maxPreviewWidth = displaySize.y;
        maxPreviewHeight = displaySize.x;
      }

      if (maxPreviewWidth > MAX_PREVIEW_WIDTH) {
        maxPreviewWidth = MAX_PREVIEW_WIDTH;
      }

      if (maxPreviewHeight > MAX_PREVIEW_HEIGHT) {
        maxPreviewHeight = MAX_PREVIEW_HEIGHT;
      }

      // Danger, W.R.! Attempting to use too large a preview size could  exceed the camera
      // bus' bandwidth limitation, resulting in gorgeous previews but the storage of
      // garbage capture data.
      previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class),
              rotatedPreviewWidth, rotatedPreviewHeight, maxPreviewWidth,
              maxPreviewHeight, largest);



      synchronized (mCameraStateLock) {
        // Set up ImageReaders for JPEG and RAW outputs.  Place these in a reference
        // counted wrapper to ensure they are only closed when all background tasks
        // using them are finished.
        if (jpegImageReader == null || jpegImageReader.getAndRetain() == null) {
          jpegImageReader = new RefCountedAutoCloseable<>(
                  ImageReader.newInstance(previewSize.getWidth(),
                          previewSize.getHeight(), ImageFormat.YUV_420_888, /*maxImages*/5));
        }


        this.characteristics = characteristics;

      }


      // We fit the aspect ratio of TextureView to the size of preview we picked.
      final int orientation = getResources().getConfiguration().orientation;
      if (orientation == Configuration.ORIENTATION_LANDSCAPE) {
        textureView.setAspectRatio(previewSize.getWidth(), previewSize.getHeight());
      } else {
        textureView.setAspectRatio(previewSize.getHeight(), previewSize.getWidth());
      }
    } catch (final CameraAccessException e) {
      LOGGER.e(e, "Exception!");
    } catch (final NullPointerException e) {
      // Currently an NPE is thrown when the Camera2API is used but not supported on the
      // device this code runs.
      // TODO(andrewharp): abstract ErrorDialog/RuntimeException handling out into new method and
      // reuse throughout app.
      ErrorDialog.newInstance(getString(R.string.camera_error))
              .show(getChildFragmentManager(), FRAGMENT_DIALOG);
      throw new RuntimeException(getString(R.string.camera_error));
    }

    cameraConnectionCallback.onPreviewSizeChosen(previewSize, sensorOrientation);
  }






  static {
    ORIENTATIONS.append(Surface.ROTATION_0, 0);
    ORIENTATIONS.append(Surface.ROTATION_90, 90);
    ORIENTATIONS.append(Surface.ROTATION_180, 180);
    ORIENTATIONS.append(Surface.ROTATION_270, 270);
  }

  /**
   * Request code for camera permissions.
   */
  private static final int REQUEST_CAMERA_PERMISSIONS = 1;

  /**
   * Permissions required to take a picture.
   */
  private static final String[] CAMERA_PERMISSIONS = {
          Manifest.permission.CAMERA,
          Manifest.permission.READ_EXTERNAL_STORAGE,
          Manifest.permission.WRITE_EXTERNAL_STORAGE,
  };

  /**
   * Timeout for the pre-capture sequence.
   */
  private static final long PRECAPTURE_TIMEOUT_MS = 1000;

  /**
   * Tolerance when comparing aspect ratios.
   */
  private static final double ASPECT_RATIO_TOLERANCE = 0.005;




  /**
   * An {@link OrientationEventListener} used to determine when device rotation has occurred.
   * This is mainly necessary for when the device is rotated by 180 degrees, in which case
   * onCreate or onConfigurationChanged is not called as the view dimensions remain the same,
   * but the orientation of the has changed, and thus the preview rotation must be updated.
   */
  private OrientationEventListener orientationListener;

  /**
   * {@link TextureView.SurfaceTextureListener} handles several lifecycle events of a
   * {@link TextureView}.
   */
  private final TextureView.SurfaceTextureListener surfaceTextureListener
          = new TextureView.SurfaceTextureListener() {

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
      openCamera(width, height);
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) {
      configureTransform(width, height);
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
      synchronized (mCameraStateLock) {
        previewSize = null;
      }
      return true;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture texture) {
    }

  };

  /**
   * A counter for tracking corresponding {@link CaptureRequest}s and {@link CaptureResult}s
   * across the {@link CameraCaptureSession} capture callbacks.
   */
  private final AtomicInteger requestCounter = new AtomicInteger();



  /**
   * A lock protecting camera state.
   */


  private final Object mCameraStateLock = new Object();




  /**
   * The {@link CameraCharacteristics} for the currently configured camera device.
   */
  private CameraCharacteristics characteristics;



  /**
   * A reference counted holder wrapping the {@link ImageReader} that handles JPEG image
   * captures. This is used to allow us to clean up the {@link ImageReader} when all background
   * tasks using its {@link Image}s have completed.
   */
  private RefCountedAutoCloseable<ImageReader> jpegImageReader;




  /**
   * Whether or not the currently configured camera device is fixed-focus.
   */
  private boolean mNoAFRun = false;

  /**
   * Number of pending user requests to capture a photo.
   */
  private int mPendingUserCaptures = 0;

  /**
   * Request ID to {@link ImageSaver.ImageSaverBuilder} mapping for in-progress JPEG captures.
   */
  private final TreeMap<Integer, ImageSaver.ImageSaverBuilder> mJpegResultQueue = new TreeMap<>();




  /**
   * The state of the camera device.
   *
   * @see #mPreCaptureCallback
   */
  private int mState = STATE_CLOSED;

  /**
   * Timer to use with pre-capture sequence to ensure a timely capture if 3A convergence is
   * taking too long.
   */
  private long mCaptureTimer;

  //**********************************************************************************************

  /**
   * {@link CameraDevice.StateCallback} is called when the currently active {@link CameraDevice}
   * changes its state.
   */

  // mudar dps
  private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    public void onOpened(CameraDevice cd) {
      // This method is called when the camera is opened.  We start camera preview here if
      // the TextureView displaying this has been set up.
      synchronized (mCameraStateLock) {
        mState = STATE_OPENED;
        cameraOpenCloseLock.release();
        cameraDevice = cd;

        // Start the preview session if the TextureView has been set up already.
        if (previewSize != null && textureView.isAvailable()) {
          createCameraPreviewSession();
        }
      }
    }

    @Override
    public void onDisconnected(CameraDevice cd) {
      synchronized (mCameraStateLock) {
        mState = STATE_CLOSED;
        cameraOpenCloseLock.release();
        cd.close();
        cameraDevice = null;
      }
    }

    @Override
    public void onError(CameraDevice cd, int error) {
      Log.e(TAG, "Received camera device error: " + error);
      synchronized (mCameraStateLock) {
        mState = STATE_CLOSED;
        cameraOpenCloseLock.release();
        cd.close();
        cameraDevice = null;
      }
      Activity activity = getActivity();
      if (null != activity) {
        activity.finish();
      }
    }

  };



  public Integer getIso(){
    return iso;
  }

  /**
   * A {@link CameraCaptureSession.CaptureCallback} that handles events for the preview and
   * pre-capture sequence.
   */
  private CameraCaptureSession.CaptureCallback mPreCaptureCallback
          = new CameraCaptureSession.CaptureCallback() {

    private void process(CaptureResult result) {
      synchronized (mCameraStateLock) {
        switch (mState) {
          case STATE_PREVIEW: {
            // We have nothing to do when the camera preview is running normally.
            break;
          }
          case STATE_WAITING_FOR_3A_CONVERGENCE: {
            boolean readyToCapture = true;
            if (!mNoAFRun) {
              Integer afState = result.get(CaptureResult.CONTROL_AF_STATE);

              if (afState == null) {
                break;
              }

              // If auto-focus has reached locked state, we are ready to capture
              readyToCapture =
                      (afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                              afState == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED);
            }

            iso = result.get(CaptureResult.SENSOR_SENSITIVITY);

            // If we are running on an non-legacy device, we should also wait until
            // auto-exposure and auto-white-balance have converged as well before
            // taking a picture.
            if (!isLegacyLocked()) {
              Integer aeState = result.get(CaptureResult.CONTROL_AE_STATE);
              Integer awbState = result.get(CaptureResult.CONTROL_AWB_STATE);
              if (aeState == null || awbState == null) {
                break;
              }

              readyToCapture = readyToCapture &&
                      aeState == CaptureResult.CONTROL_AE_STATE_CONVERGED &&
                      awbState == CaptureResult.CONTROL_AWB_STATE_CONVERGED;
            }

            // If we haven't finished the pre-capture sequence but have hit our maximum
            // wait timeout, too bad! Begin capture anyway.
            if (!readyToCapture && hitTimeoutLocked() && isoTest == false) {
              Log.w(TAG, "Timed out waiting for pre-capture sequence to complete.");
              readyToCapture = true;
            }

            if (readyToCapture && mPendingUserCaptures > 0) {
              // Capture once for each user tap of the "Picture" button.
              while (mPendingUserCaptures > 0) {
                captureStillPictureLocked();
                mPendingUserCaptures--;
              }
              // After this, the camera will go back to the normal state of preview.
              mState = STATE_PREVIEW;
            }
          }
        }



      }

    }

    @Override
    public void onCaptureProgressed(CameraCaptureSession session, CaptureRequest request,
                                    CaptureResult partialResult) {
      process(partialResult);

    }

    @Override
    public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request,
                                   TotalCaptureResult result) {
      process(result);

    }

  };


  private final CameraCaptureSession.CaptureCallback captureCallback
          = new CameraCaptureSession.CaptureCallback() {
    @Override
    public void onCaptureStarted(CameraCaptureSession session, CaptureRequest request,
                                 long timestamp, long frameNumber) {
      String currentDateTime = generateTimestamp();

      File jpegFile = new File(Environment.
              getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM),
              "CAAI_" + currentDateTime + ".jpg");

      // Look up the ImageSaverBuilder for this request and update it with the file name
      // based on the capture start time.
      ImageSaver.ImageSaverBuilder jpegBuilder;

      int requestId = (int) request.getTag();
      synchronized (mCameraStateLock) {
        jpegBuilder = mJpegResultQueue.get(requestId);

      }

      if (jpegBuilder != null) jpegBuilder.setFile(jpegFile);

    }

    @Override
    public void onCaptureProgressed(
            final CameraCaptureSession session,
            final CaptureRequest request,
            final CaptureResult partialResult) {}
    @Override
    public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request,
                                   TotalCaptureResult result) {

      int requestId = (int) request.getTag();
      ImageSaver.ImageSaverBuilder jpegBuilder;

      StringBuilder sb = new StringBuilder();

      // Look up the ImageSaverBuilder for this request and update it with the CaptureResult
      synchronized (mCameraStateLock) {
        jpegBuilder = mJpegResultQueue.get(requestId);


        if (jpegBuilder != null) {
          jpegBuilder.setResult(result);

          sb.append("Saving JPEG as: ");
          sb.append(jpegBuilder.getSaveLocation());
        }


        // If we have all the results necessary, save the image to a file in the background.
        handleCompletionLocked(requestId, jpegBuilder, mJpegResultQueue);

        finishedCaptureLocked();

      }

    }

    @Override
    public void onCaptureFailed(CameraCaptureSession session, CaptureRequest request,
                                CaptureFailure failure) {
      int requestId = (int) request.getTag();
      synchronized (mCameraStateLock) {
        mJpegResultQueue.remove(requestId);

        finishedCaptureLocked();
      }


    }

  };




  @Override
  public void onViewCreated(final View view, Bundle savedInstanceState) {

    textureView = (AutoFitTextureView) view.findViewById(R.id.texture);

    // Setup a new OrientationEventListener.  This is used to handle rotation events like a
    // 180 degree rotation that do not normally trigger a call to onCreate to do view re-layout
    // or otherwise cause the preview TextureView's size to change.
    orientationListener = new OrientationEventListener(getActivity(),
            SensorManager.SENSOR_DELAY_NORMAL) {
      @RequiresApi(api = Build.VERSION_CODES.M)
      @Override
      public void onOrientationChanged(int orientation) {
        if (textureView != null && textureView.isAvailable()) {
          configureTransform(textureView.getWidth(), textureView.getHeight());
        }
      }
    };
  }

  @RequiresApi(api = Build.VERSION_CODES.M)
  @Override
  public void onResume() {
    super.onResume();
    startBackgroundThread();


    // When the screen is turned off and turned back on, the SurfaceTexture is already
    // available, and "onSurfaceTextureAvailable" will not be called. In that case, we should
    // configure the preview bounds here (otherwise, we wait until the surface is ready in
    // the SurfaceTextureListener).
    if (textureView.isAvailable()) {
      configureTransform(textureView.getWidth(), textureView.getHeight());
      openCamera(textureView.getWidth(), textureView.getHeight());
    } else {
      textureView.setSurfaceTextureListener(surfaceTextureListener);
    }
    if (orientationListener != null && orientationListener.canDetectOrientation()) {
      orientationListener.enable();
    }
  }

  @Override
  public void onPause() {
    if (orientationListener != null) {
      orientationListener.disable();
    }
    closeCamera();
    stopBackgroundThread();
    super.onPause();
  }

  @Override
  public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
    if (requestCode == REQUEST_CAMERA_PERMISSIONS) {
      for (int result : grantResults) {
        if (result != PackageManager.PERMISSION_GRANTED) {
          showMissingPermissionError();
          return;
        }
      }
    } else {
      super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
  }



  /**
   * Opens the camera specified by {@link #cameraId}.
   */
  @RequiresApi(api = Build.VERSION_CODES.M)
  @SuppressWarnings("MissingPermission")
  private boolean openCamera(final int width, final int height) {
    setUpCameraOutputs( width, height);

    if (!hasAllPermissionsGranted()) {
      requestCameraPermissions();
      return false;
    }

    configureTransform(width, height);
    Activity activity = getActivity();

    CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
    try {
      // Wait for any previously running session to finish.
      if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
        throw new RuntimeException("Time out waiting to lock camera opening.");
      }

      String cameraId;
      Handler backgroundHandler;
      synchronized (mCameraStateLock) {
        cameraId = this.cameraId;
        backgroundHandler = this.backgroundHandler;
      }

      // Attempt to open the camera. mStateCallback will be called on the background handler's
      // thread when this succeeds or fails.
      manager.openCamera(cameraId, mStateCallback, backgroundHandler);
      return true;
    } catch (CameraAccessException e) {
      e.printStackTrace();
    } catch (InterruptedException e) {
      throw new RuntimeException("Interrupted while trying to lock camera opening.", e);
    }
    return false;
  }

  /**
   * Requests permissions necessary to use camera and save pictures.
   */
  private void requestCameraPermissions() {
    if (shouldShowRationale()) {
      PermissionConfirmationDialog.newInstance().show(getChildFragmentManager(), "dialog");
    } else {
      ActivityCompat.requestPermissions(getActivity(), CAMERA_PERMISSIONS, REQUEST_CAMERA_PERMISSIONS);
    }
  }

  /**
   * Tells whether all the necessary permissions are granted to this app.
   *
   * @return True if all the required permissions are granted.
   */
  private boolean hasAllPermissionsGranted() {
    for (String permission : CAMERA_PERMISSIONS) {
      if (ActivityCompat.checkSelfPermission(getActivity(), permission)
              != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  /**
   * Gets whether you should show UI with rationale for requesting the permissions.
   *
   * @return True if the UI should be shown.
   */
  private boolean shouldShowRationale() {
    for (String permission : CAMERA_PERMISSIONS) {
      if (ActivityCompat.shouldShowRequestPermissionRationale(getActivity(), permission)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Shows that this app really needs the permission and finishes the app.
   */
  private void showMissingPermissionError() {
    Activity activity = getActivity();
    if (activity != null) {
      Toast.makeText(activity, R.string.request_permission, Toast.LENGTH_SHORT).show();
      activity.finish();
    }
  }





  /**
   * Configure the given {@link CaptureRequest.Builder} to use auto-focus, auto-exposure, and
   * auto-white-balance controls if available.
   * <p/>
   * Call this only with {@link #mCameraStateLock} held.
   *
   * @param builder the builder to configure.
   */
  private void setup3AControlsLocked(CaptureRequest.Builder builder) {
    // Enable auto-magical 3A run by camera device
    builder.set(CaptureRequest.CONTROL_MODE,
            CaptureRequest.CONTROL_MODE_AUTO);

    Float minFocusDist =
            characteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE);

    // If MINIMUM_FOCUS_DISTANCE is 0, lens is fixed-focus and we need to skip the AF run.
    mNoAFRun = (minFocusDist == null || minFocusDist == 0);

    if (!mNoAFRun) {
      // If there is a "continuous picture" mode available, use it, otherwise default to AUTO.
      if (contains(characteristics.get(
              CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES),
              CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)) {
        builder.set(CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
      } else {
        builder.set(CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_AUTO);
      }
    }


    // If there is an auto-magical white balance control mode available, use it.
    if (contains(characteristics.get(
            CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES),
            CaptureRequest.CONTROL_AWB_MODE_AUTO)) {
      // Allow AWB to run auto-magically if this device supports this
      builder.set(CaptureRequest.CONTROL_AWB_MODE,
              CaptureRequest.CONTROL_AWB_MODE_AUTO);
    }
  }




  /**
   * Initiate a still image capture.
   * <p/>
   * This function sends a capture request that initiates a pre-capture sequence in our state
   * machine that waits for auto-focus to finish, ending in a "locked" state where the lens is no
   * longer moving, waits for auto-exposure to choose a good exposure value, and waits for
   * auto-white-balance to converge.
   */

  public void takePicture() {




    synchronized (mCameraStateLock) {
      mPendingUserCaptures++;

      // If we already triggered a pre-capture sequence, or are in a state where we cannot
      // do this, return immediately.
      if (mState != STATE_PREVIEW) {
        return;
      }

      try {
        // Trigger an auto-focus run if camera is capable. If the camera is already focused,
        // this should do nothing.
        if (!mNoAFRun) {
          previewRequestBuilder.set(CaptureRequest.CONTROL_AF_TRIGGER,
                  CameraMetadata.CONTROL_AF_TRIGGER_START);
        }

        // If this is not a legacy device, we can also trigger an auto-exposure metering
        // run.
        if (!isLegacyLocked()) {
          // Tell the camera to lock focus.
          previewRequestBuilder.set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER,
                  CameraMetadata.CONTROL_AE_PRECAPTURE_TRIGGER_START);
        }

        // Update state machine to wait for auto-focus, auto-exposure, and
        // auto-white-balance (aka. "3A") to converge.
        mState = STATE_WAITING_FOR_3A_CONVERGENCE;

        // Start a timer for the pre-capture sequence.
        startTimerLocked();

        // Replace the existing repeating request with one with updated 3A triggers.
        captureSession.capture(previewRequestBuilder.build(), mPreCaptureCallback,
                backgroundHandler);
      } catch (CameraAccessException e) {
        e.printStackTrace();
      }
    }


  }


  public void seeObjects() throws CameraAccessException {
    mState = STATE_WAITING_FOR_3A_CONVERGENCE;
    isoTest = true;
    jpegImageReader.get().setOnImageAvailableListener(imageListener, backgroundHandler);

    // This is the CaptureRequest.Builder that we use to take a picture.
    final CaptureRequest.Builder captureBuilder =
            cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);


    captureBuilder.addTarget(jpegImageReader.get().getSurface());

    // Set request tag to easily track results in callbacks.
    captureBuilder.setTag(requestCounter.getAndIncrement());

    CaptureRequest request = captureBuilder.build();


    captureSession.capture(request, mPreCaptureCallback, backgroundHandler);

    final Handler handler = new Handler();
    handler.postDelayed(() -> {
    mState = STATE_PREVIEW;
      //delay 1s
    }, 1000);
    Log.e(TAG, "ISO:" + iso );
    final Handler handler2 = new Handler();
    handler.postDelayed(() -> {
      if(iso != null){
 if(iso >= 3200){
      String toError = "It's a little dark";
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
        Toast.makeText(getContext(), toError, Toast.LENGTH_SHORT);
      }

      t2.speak(toError, TextToSpeech.QUEUE_FLUSH, null);
 }
        if(iso <= 300){
          String toError = "It's too light!";
          if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Toast.makeText(getContext(), toError, Toast.LENGTH_SHORT);
          }

          t2.speak(toError, TextToSpeech.QUEUE_FLUSH, null);
        }
      }
    }, 500);
  }

  /**
   * Send a capture request to the camera device that initiates a capture targeting the JPEG and
   * RAW outputs.
   * <p/>
   * Call this only with {@link #mCameraStateLock} held.
   */
  private void captureStillPictureLocked() {

    jpegImageReader.get().setOnImageAvailableListener(mOnJpegImageAvailableListener, backgroundHandler);
    try {
      final Activity activity = getActivity();
      if (null == activity || null == cameraDevice) {
        return;
      }
      // This is the CaptureRequest.Builder that we use to take a picture.
      final CaptureRequest.Builder captureBuilder =
              cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);

      captureBuilder.addTarget(jpegImageReader.get().getSurface());





      // Use the same AE and AF modes as the preview.
      setup3AControlsLocked(captureBuilder);

      // Set orientation.
      int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
      captureBuilder.set(CaptureRequest.JPEG_ORIENTATION,
              sensorToDeviceRotation(characteristics, rotation));

      // Set request tag to easily track results in callbacks.
      captureBuilder.setTag(requestCounter.getAndIncrement());

      CaptureRequest request = captureBuilder.build();

      // Create an ImageSaverBuilder in which to collect results, and add it to the queue
      // of active requests.
      ImageSaver.ImageSaverBuilder jpegBuilder = new ImageSaver.ImageSaverBuilder(activity)
              .setCharacteristics(characteristics);


      mJpegResultQueue.put((int) request.getTag(), jpegBuilder);


      captureSession.capture(request, captureCallback, backgroundHandler);

    } catch (CameraAccessException e) {
      e.printStackTrace();
    }
  }


  private void finishedCaptureLocked() {
    try {
      // Reset the auto-focus trigger in case AF didn't run quickly enough.
      if (!mNoAFRun) {
        previewRequestBuilder.set(CaptureRequest.CONTROL_AF_TRIGGER,
                CameraMetadata.CONTROL_AF_TRIGGER_CANCEL);

        captureSession.capture(previewRequestBuilder.build(), mPreCaptureCallback,
                backgroundHandler);

        previewRequestBuilder.set(CaptureRequest.CONTROL_AF_TRIGGER,
                CameraMetadata.CONTROL_AF_TRIGGER_IDLE);
      }
    } catch (CameraAccessException e) {
      e.printStackTrace();
    }
  }

  /**
   * Retrieve the next {@link Image} from a reference counted {@link ImageReader}, retaining
   * that {@link ImageReader} until that {@link Image} is no longer in use, and set this
   * {@link Image} as the result for the next request in the queue of pending requests.  If
   * all necessary information is available, begin saving the image to a file in a background
   * thread.
   *
   * @param pendingQueue the currently active requests.
   * @param reader       a reference counted wrapper containing an {@link ImageReader} from which
   *                     to acquire an image.
   */
  private void dequeueAndSaveImage(TreeMap<Integer, ImageSaver.ImageSaverBuilder> pendingQueue,
                                   RefCountedAutoCloseable<ImageReader> reader) {
    synchronized (mCameraStateLock) {
      Map.Entry<Integer, ImageSaver.ImageSaverBuilder> entry =
              pendingQueue.firstEntry();
      ImageSaver.ImageSaverBuilder builder = entry.getValue();

      // Increment reference count to prevent ImageReader from being closed while we
      // are saving its Images in a background thread (otherwise their resources may
      // be freed while we are writing to a file).
      if (reader == null || reader.getAndRetain() == null) {
        Log.e(TAG, "Paused the activity before we could save the image," +
                " ImageReader already closed.");
        pendingQueue.remove(entry.getKey());
        return;
      }

      Image image;
      try {
        image = reader.get().acquireNextImage();
      } catch (IllegalStateException e) {
        Log.e(TAG, "Too many images queued for saving, dropping image for request: " +
                entry.getKey());
        pendingQueue.remove(entry.getKey());
        return;
      }

      if(image != null) {
        builder.setRefCountedReader(reader).setImage(image);
        showToast("Imagem Salva! ");
      }else{
        Log.e(TAG, "Image null, try again!" );
      }
      handleCompletionLocked(entry.getKey(), builder, pendingQueue);
    }
  }

  /**
   * Runnable that saves an {@link Image} into the specified {@link File}, and updates
   * {@link android.provider.MediaStore} to include the resulting file.
   * <p/>
   * This can be constructed through an {@link ImageSaverBuilder} as the necessary image and
   * result information becomes available.
   */
  private static class ImageSaver implements Runnable {

    /**
     * The image to save.
     */
    private final Image mImage;
    /**
     * The file we save the image into.
     */
    private final File mFile;

    /**
     * The CaptureResult for this image capture.
     */
    private final CaptureResult mCaptureResult;

    /**
     * The CameraCharacteristics for this camera device.
     */
    private final CameraCharacteristics mCharacteristics;

    /**
     * The Context to use when updating MediaStore with the saved images.
     */
    private final Context mContext;

    /**
     * A reference counted wrapper for the ImageReader that owns the given image.
     */
    private final RefCountedAutoCloseable<ImageReader> mReader;

    private ImageSaver(Image image, File file, CaptureResult result,
                       CameraCharacteristics characteristics, Context context,
                       RefCountedAutoCloseable<ImageReader> reader) {
      mImage = image;
      mFile = file;
      mCaptureResult = result;
      mCharacteristics = characteristics;
      mContext = context;
      mReader = reader;
    }

    @Override
    public void run() {
      boolean success = false;
      byte[] data = NV21toJPEG(YUV420toNV21(mImage), mImage.getWidth(), mImage.getHeight(), 80);



      FileOutputStream output = null;
      try {
        output = new FileOutputStream(mFile);
        output.write(data);
        success = true;
      } catch (IOException e) {
        e.printStackTrace();
      } finally {
        mImage.close();
        closeOutput(output);
      }






      // Decrement reference count to allow ImageReader to be closed to free up resources.
      mReader.close();

      // If saving the file succeeded, update MediaStore.
      if (success) {
        MediaScannerConnection.scanFile(mContext, new String[]{mFile.getPath()},
                /*mimeTypes*/null, new MediaScannerConnection.MediaScannerConnectionClient() {
                  @Override
                  public void onMediaScannerConnected() {
                    // Do nothing
                  }

                  @Override
                  public void onScanCompleted(String path, Uri uri) {
                    Log.i(TAG, "Scanned " + path + ":");
                    Log.i(TAG, "-> uri=" + uri);
                  }
                });
      }
    }


    private static byte[] NV21toJPEG(byte[] nv21, int width, int height, int quality) {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      YuvImage yuv = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
      yuv.compressToJpeg(new Rect(0, 0, width, height), quality, out);
      return out.toByteArray();
    }

    private static byte[] YUV420toNV21(Image image) {
      Rect crop = image.getCropRect();
      int format = image.getFormat();
      int width = crop.width();
      int height = crop.height();
      Image.Plane[] planes = image.getPlanes();
      byte[] data = new byte[width * height * ImageFormat.getBitsPerPixel(format) / 8];
      byte[] rowData = new byte[planes[0].getRowStride()];

      int channelOffset = 0;
      int outputStride = 1;
      for (int i = 0; i < planes.length; i++) {
        switch (i) {
          case 0:
            channelOffset = 0;
            outputStride = 1;
            break;
          case 1:
            channelOffset = width * height + 1;
            outputStride = 2;
            break;
          case 2:
            channelOffset = width * height;
            outputStride = 2;
            break;
        }

        ByteBuffer buffer = planes[i].getBuffer();
        int rowStride = planes[i].getRowStride();
        int pixelStride = planes[i].getPixelStride();

        int shift = (i == 0) ? 0 : 1;
        int w = width >> shift;
        int h = height >> shift;
        buffer.position(rowStride * (crop.top >> shift) + pixelStride * (crop.left >> shift));
        for (int row = 0; row < h; row++) {
          int length;
          if (pixelStride == 1 && outputStride == 1) {
            length = w;
            buffer.get(data, channelOffset, length);
            channelOffset += length;
          } else {
            length = (w - 1) * pixelStride + 1;
            buffer.get(rowData, 0, length);
            for (int col = 0; col < w; col++) {
              data[channelOffset] = rowData[col * pixelStride];
              channelOffset += outputStride;
            }
          }
          if (row < h - 1) {
            buffer.position(buffer.position() + rowStride - length);
          }
        }
      }
      return data;
    }

    /**
     * Builder class for constructing {@link ImageSaver}s.
     * <p/>
     * This class is thread safe.
     */
    public static class ImageSaverBuilder {
      private Image mImage;
      private File mFile;
      private CaptureResult mCaptureResult;
      private CameraCharacteristics mCharacteristics;
      private Context mContext;
      private RefCountedAutoCloseable<ImageReader> mReader;

      /**
       * Construct a new ImageSaverBuilder using the given {@link Context}.
       *
       * @param context a {@link Context} to for accessing the
       *                {@link android.provider.MediaStore}.
       */
      public ImageSaverBuilder(final Context context) {
        mContext = context;
      }

      public synchronized ImageSaverBuilder setRefCountedReader(
              RefCountedAutoCloseable<ImageReader> reader) {
        if (reader == null) throw new NullPointerException();

        mReader = reader;
        return this;
      }

      public synchronized ImageSaverBuilder setImage(final Image image) {
        if (image == null) throw new NullPointerException();
        mImage = image;
        return this;
      }

      public synchronized ImageSaverBuilder setFile(final File file) {
        if (file == null) throw new NullPointerException();
        mFile = file;
        return this;
      }

      public synchronized ImageSaverBuilder setResult(final CaptureResult result) {
        if (result == null) throw new NullPointerException();
        mCaptureResult = result;
        return this;
      }

      public synchronized ImageSaverBuilder setCharacteristics(
              final CameraCharacteristics characteristics) {
        if (characteristics == null) throw new NullPointerException();
        mCharacteristics = characteristics;
        return this;
      }

      public synchronized ImageSaver buildIfComplete() {
        if (!isComplete()) {
          return null;
        }
        return new ImageSaver(mImage, mFile, mCaptureResult, mCharacteristics, mContext,
                mReader);
      }

      public synchronized String getSaveLocation() {
        return (mFile == null) ? "Unknown" : mFile.toString();
      }

      private boolean isComplete() {
        return mImage != null && mFile != null && mCaptureResult != null
                && mCharacteristics != null;
      }
    }
  }

  // Utility classes and methods:
  // *********************************************************************************************



  /**
   * A wrapper for an {@link AutoCloseable} object that implements reference counting to allow
   * for resource management.
   */
  public static class RefCountedAutoCloseable<T extends AutoCloseable> implements AutoCloseable {
    private T mObject;
    private long mRefCount = 0;

    /**
     * Wrap the given object.
     *
     * @param object an object to wrap.
     */
    public RefCountedAutoCloseable(T object) {
      if (object == null) throw new NullPointerException();
      mObject = object;
    }

    /**
     * Increment the reference count and return the wrapped object.
     *
     * @return the wrapped object, or null if the object has been released.
     */
    public synchronized T getAndRetain() {
      if (mRefCount < 0) {
        return null;
      }
      mRefCount++;
      return mObject;
    }

    /**
     * Return the wrapped object.
     *
     * @return the wrapped object, or null if the object has been released.
     */
    public synchronized T get() {
      return mObject;
    }

    /**
     * Decrement the reference count and release the wrapped object if there are no other
     * users retaining this object.
     */
    @Override
    public synchronized void close() {
      if (mRefCount >= 0) {
        mRefCount--;
        if (mRefCount < 0) {
          try {
            mObject.close();
          } catch (Exception e) {
            throw new RuntimeException(e);
          } finally {
            mObject = null;
          }
        }
      }
    }
  }

  /**
   * Given {@code choices} of {@code Size}s supported by a camera, choose the smallest one that
   * is at least as large as the respective texture view size, and that is at most as large as the
   * respective max size, and whose aspect ratio matches with the specified value. If such size
   * doesn't exist, choose the largest one that is at most as large as the respective max size,
   * and whose aspect ratio matches with the specified value.
   *
   * @param choices           The list of sizes that the camera supports for the intended output
   *                          class
   * @param textureViewWidth  The width of the texture view relative to sensor coordinate
   * @param textureViewHeight The height of the texture view relative to sensor coordinate
   * @param maxWidth          The maximum width that can be chosen
   * @param maxHeight         The maximum height that can be chosen
   * @param aspectRatio       The aspect ratio
   * @return The optimal {@code Size}, or an arbitrary one if none were big enough
   */
  private static Size chooseOptimalSize(Size[] choices, int textureViewWidth,
                                        int textureViewHeight, int maxWidth, int maxHeight, Size aspectRatio) {
    // Collect the supported resolutions that are at least as big as the preview Surface
    List<Size> bigEnough = new ArrayList<>();
    // Collect the supported resolutions that are smaller than the preview Surface
    List<Size> notBigEnough = new ArrayList<>();
    int w = aspectRatio.getWidth();
    int h = aspectRatio.getHeight();
    for (Size option : choices) {
      if (option.getWidth() <= maxWidth && option.getHeight() <= maxHeight &&
              option.getHeight() == option.getWidth() * h / w) {
        if (option.getWidth() >= textureViewWidth &&
                option.getHeight() >= textureViewHeight) {
          bigEnough.add(option);
        } else {
          notBigEnough.add(option);
        }
      }
    }

    // Pick the smallest of those big enough. If there is no one big enough, pick the
    // largest of those not big enough.
    if (bigEnough.size() > 0) {
      return Collections.min(bigEnough, new CompareSizesByArea());
    } else if (notBigEnough.size() > 0) {
      return Collections.max(notBigEnough, new CompareSizesByArea());
    } else {
      Log.e(TAG, "Couldn't find any suitable preview size");
      return choices[0];
    }
  }

  /**
   * Generate a string containing a formatted timestamp with the current date and time.
   *
   * @return a {@link String} representing a time.
   */
  private static String generateTimestamp() {
    SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss_SSS", Locale.US);
    return sdf.format(new Date());
  }

  /**
   * Cleanup the given {@link OutputStream}.
   *
   * @param outputStream the stream to close.
   */
  private static void closeOutput(OutputStream outputStream) {
    if (null != outputStream) {
      try {
        outputStream.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  /**
   * Return true if the given array contains the given integer.
   *
   * @param modes array to check.
   * @param mode  integer to get for.
   * @return true if the array contains the given integer, otherwise false.
   */
  private static boolean contains(int[] modes, int mode) {
    if (modes == null) {
      return false;
    }
    for (int i : modes) {
      if (i == mode) {
        return true;
      }
    }
    return false;
  }

  /**
   * Return true if the two given {@link Size}s have the same aspect ratio.
   *
   * @param a first {@link Size} to compare.
   * @param b second {@link Size} to compare.
   * @return true if the sizes have the same aspect ratio, otherwise false.
   */
  private static boolean checkAspectsEqual(Size a, Size b) {
    double aAspect = a.getWidth() / (double) a.getHeight();
    double bAspect = b.getWidth() / (double) b.getHeight();
    return Math.abs(aAspect - bAspect) <= ASPECT_RATIO_TOLERANCE;
  }

  /**
   * Rotation need to transform from the camera sensor orientation to the device's current
   * orientation.
   *
   * @param c                 the {@link CameraCharacteristics} to query for the camera sensor
   *                          orientation.
   * @param deviceOrientation the current device orientation relative to the native device
   *                          orientation.
   * @return the total rotation from the sensor orientation to the current device orientation.
   */
  private static int sensorToDeviceRotation(CameraCharacteristics c, int deviceOrientation) {
    int sensorOrientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION);

    // Get device orientation in degrees
    deviceOrientation = ORIENTATIONS.get(deviceOrientation);

    // Reverse device orientation for front-facing cameras
    if (c.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT) {
      deviceOrientation = -deviceOrientation;
    }

    // Calculate desired JPEG orientation relative to camera orientation to make
    // the image upright relative to the device orientation
    return (sensorOrientation - deviceOrientation + 360) % 360;
  }


  /**
   * If the given request has been completed, remove it from the queue of active requests and
   * send an {@link ImageSaver} with the results from this request to a background thread to
   * save a file.
   * <p/>
   * Call this only with {@link #mCameraStateLock} held.
   *
   * @param requestId the ID of the {@link CaptureRequest} to handle.
   * @param builder   the {@link ImageSaver.ImageSaverBuilder} for this request.
   * @param queue     the queue to remove this request from, if completed.
   */
  private void handleCompletionLocked(int requestId, ImageSaver.ImageSaverBuilder builder,
                                      TreeMap<Integer, ImageSaver.ImageSaverBuilder> queue) {
    if (builder == null) return;
    ImageSaver saver = builder.buildIfComplete();
    if (saver != null) {
      queue.remove(requestId);
      AsyncTask.THREAD_POOL_EXECUTOR.execute(saver);
    }
  }

  /**
   * Check if we are using a device that only supports the LEGACY hardware level.
   * <p/>
   * Call this only with {@link #mCameraStateLock} held.
   *
   * @return true if this is a legacy device.
   */
  private boolean isLegacyLocked() {
    return characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL) ==
            CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY;
  }

  /**
   * Start the timer for the pre-capture sequence.
   * <p/>
   * Call this only with {@link #mCameraStateLock} held.
   */
  private void startTimerLocked() {
    mCaptureTimer = SystemClock.elapsedRealtime();
  }

  /**
   * Check if the timer for the pre-capture sequence has been hit.
   * <p/>
   * Call this only with {@link #mCameraStateLock} held.
   *
   * @return true if the timeout occurred.
   */
  private boolean hitTimeoutLocked() {
    return (SystemClock.elapsedRealtime() - mCaptureTimer) > PRECAPTURE_TIMEOUT_MS;
  }

  /**
   * A dialog that explains about the necessary permissions.
   */
  public static class PermissionConfirmationDialog extends DialogFragment {

    public static PermissionConfirmationDialog newInstance() {
      return new PermissionConfirmationDialog();
    }

    @Override
    public Dialog onCreateDialog(Bundle savedInstanceState) {

      return new AlertDialog.Builder(getActivity())
              .setMessage(R.string.request_permission)
              .setPositiveButton(android.R.string.ok, (dialog, which) -> ActivityCompat.requestPermissions(getActivity(), CAMERA_PERMISSIONS,
                      REQUEST_CAMERA_PERMISSIONS))
              .setNegativeButton(android.R.string.cancel,
                      (dialog, which) -> getActivity().finish())
              .create();
    }

  }

}