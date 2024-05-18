package com.example.fabricstaindetector;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.fabricstaindetector.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    private ImageView imgView;
    private Button select, predict;
    private TextView tv;
    private Bitmap img;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgView = findViewById(R.id.imageView);
        tv = findViewById(R.id.textView);
        select = findViewById(R.id.button);
        predict = findViewById(R.id.button2);

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openImagePicker();
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (img != null) {
                    try {
                        ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

                        // Resize the image to match the input size of the model
                        Bitmap resizedImage = Bitmap.createScaledBitmap(img, 224, 224, true);

                        // Convert Bitmap to ByteBuffer
                        ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedImage);

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                        inputFeature0.loadBuffer(byteBuffer);

                        // Runs model inference and gets result.
                        ModelUnquant.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        // Releases model resources if no longer used.
                        model.close();

                        // Display the results
                        tv.setText("Stain Probability: " + outputFeature0.getFloatValue(0));
                    } catch (IOException e) {
                        Toast.makeText(MainActivity.this, "Error loading model: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    } catch (Exception e) {
                        Toast.makeText(MainActivity.this, "Error predicting: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                } else {
                    Toast.makeText(MainActivity.this, "Please select an image first", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void openImagePicker() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(intent, 100);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 100 && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            imgView.setImageURI(uri);
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                Toast.makeText(MainActivity.this, "Error loading image: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            }
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        int BATCH_SIZE = 1;
        int PIXEL_SIZE = 3;
        int inputSize = 224;

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE * 4);
        byteBuffer.order(java.nio.ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }
        return byteBuffer;
    }
}
