package com.example.fabricstaindetector;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

public class About extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_about);
    }

    public void clickHome(View view) {
        Intent i = new Intent(getApplicationContext(), Home.class);
        startActivity(i);
    }
}