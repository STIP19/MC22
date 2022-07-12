package com.example.blind;

import android.content.Intent;
import android.hardware.camera2.*;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.view.View;
import androidx.navigation.ui.AppBarConfiguration;
import com.example.blind.databinding.ActivityMainBinding;
import android.widget.Button;
import android.widget.EditText;


public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;

    public static int threshold = 7;
    public static int threshold2 = 12000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button button = (Button) findViewById(R.id.start_camera);

        //In final version we can start directly in the detection mode
        //startActivity(new Intent(MainActivity.this, CameraOpencv.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK|Intent.FLAG_ACTIVITY_CLEAR_TOP));

        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                EditText thresholdText = (EditText) findViewById(R.id.threshold_);
                threshold = Integer.parseInt(thresholdText.getText().toString());

                EditText threshold2Text = (EditText) findViewById(R.id.threshold2_);
                threshold2 = Integer.parseInt(threshold2Text.getText().toString());

                startActivity(new Intent(MainActivity.this, CameraOpencv.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK|Intent.FLAG_ACTIVITY_CLEAR_TOP));

            }
        });
    }
}