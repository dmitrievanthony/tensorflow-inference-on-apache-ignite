/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.ignite;

import com.sun.corba.se.impl.orbutil.ObjectWriter;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Array;
import java.net.Socket;
import java.util.Arrays;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import org.apache.commons.io.FileUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import static javax.swing.WindowConstants.DISPOSE_ON_CLOSE;

public class Client {

    static {
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String... args) throws IOException, ClassNotFoundException {
        byte[] serialized = FileUtils.readFileToByteArray(new File("/home/gridgain/tensorflow-inference-on-apache-ignite/src/main/resources/faces/saved_model.pb"));
        Graph graph = new Graph();
        graph.importGraphDef(serialized);

        Session session = new Session(graph);

        VideoCapture capture = new VideoCapture(0);

        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        JLabel label = new JLabel();
        frame.getContentPane().add(label);
        frame.pack();
        frame.setVisible(true);

        while (true) {
            Mat image = new Mat();
            if (!capture.read(image))
                throw new IllegalStateException("Something goes wrong!");

            float[] box = requestBox(image);
//            float[] box = makeMagic(session, image);
            draw(box, image);

            showResult(image, label);

            frame.pack();
            frame.setVisible(true);
        }
    }

    private static float[] requestBox(Mat image) throws IOException, ClassNotFoundException {
        long start = System.currentTimeMillis();
        float[] res = new float[4];
        try {
            Mat img = new Mat();
            Imgproc.cvtColor(image, img, Imgproc.COLOR_BGR2RGB);

            int h = img.height();
            int w = img.width();

            byte[][][][] pixels = new byte[1][h][w][3];
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++)
                    img.get(i, j, pixels[0][i][j]);
            }

            try (Socket socket = new Socket("localhost", 8765)) {
                ObjectOutputStream oos = new ObjectOutputStream(socket.getOutputStream());
                ObjectInputStream ois = new ObjectInputStream(socket.getInputStream());
                oos.writeObject(pixels);
                oos.flush();
                res = (float[])ois.readObject();

                return res;
            }
        }
        finally {
            System.out.println("Request time: " + (System.currentTimeMillis() - start) + " ms, result " + Arrays.toString(res));
        }
    }

    private static void draw(float[] box, Mat img) {
        int h = img.height();
        int w = img.width();
        Imgproc.rectangle(img, new Point(box[1] * w, box[0] * h), new Point(box[3] * w, box[2] * h), new Scalar(0, 255, 0));
    }

    public static void showResult(Mat img, JLabel label) {
        int h = img.height();
        int w = img.width();
        Imgproc.resize(img, img, new Size(w, h));
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", img, matOfByte);
        byte[] byteArr = matOfByte.toArray();
        BufferedImage bufImage;
        try {
            InputStream in = new ByteArrayInputStream(byteArr);
            bufImage = ImageIO.read(in);
            label.setIcon(new ImageIcon(bufImage));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static float[] makeMagic(Session ses, Mat srcImg) {

        Mat img = new Mat();
        Imgproc.cvtColor(srcImg, img, Imgproc.COLOR_BGR2RGB);

        int h = img.height();
        int w = img.width();

//        byte[] yuv = new byte[(int)(img.total()*img.channels())];
//        img.get(0,0,yuv);

        byte[][][][] pixels = new byte[1][h][w][3];
//        for (int i = 0; i < yuv.length; i++) {
//            int pos = i / 3;
//
//            pixels[0][pos / w][pos % w][i % 3] = yuv[i];
//        }
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++)
                img.get(i, j, pixels[0][i][j]);
        }

        Tensor image = Tensor.create(pixels, UInt8.class);
//        System.out.println("Image type: " + image.dataType());

        List<Tensor<?>> result = ses.runner()
            .feed("image_tensor:0", image)
            .fetch("detection_boxes:0")
            .fetch("detection_scores:0")
//            .fetch("detection_classes:0")
//            .fetch("num_detections:0")
            .run();

        Tensor<?> boxes = result.get(0);
        Tensor<?> scores = result.get(1);
//        Tensor<?> classes = result.get(2);
//        Tensor<?> detections = result.get(3);

        float[][][] boxesArr = new float[1][100][4];
        boxesArr = boxes.copyTo(boxesArr);
//        for (int i = 0; i < 100; i++) {
//            float[] box = boxesArr[0][i];
////            System.out.println("Box: " + Arrays.toString(box));
//        }

        float[][] scoresArr = new float[1][100];
        scoresArr = scores.copyTo(scoresArr);
//        System.out.println("Scores: " + Arrays.toString(scoresArr[0]));

        for (int i = 0; i < scoresArr[0].length; i++) {
            if (scoresArr[0][i] > 0.5)
                return boxesArr[0][i];
        }

        return new float[4];
    }
}
