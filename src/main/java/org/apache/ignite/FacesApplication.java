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

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
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

/**
 * Example of TensorFlow inference on face detection model and video stream.
 */
public class FacesApplication {

    static {
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /**
     * Starts a video capture, analyzes the inbound video stream and looking for the faces in it.
     *
     * @param args Argument.
     * @throws IOException In case of exception.
     */
    public static void main(String... args) throws IOException {
        try (Session ses = createSession("src/main/resources/models/faces/saved_model.pb")) {
            VideoCapture capture = new VideoCapture(0);
            JFrame frame = new JFrame();
            frame.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
            JLabel lb = new JLabel();
            frame.getContentPane().add(lb);
            frame.pack();
            frame.setVisible(true);

            while (true) {
                Mat image = new Mat();
                if (!capture.read(image))
                    throw new IllegalStateException("Something goes wrong!");

                List<float[]> boxes = findFace(ses, image);

                draw(boxes, image);
                showResult(image, lb);

                frame.pack();
                frame.setVisible(true);
            }
        }
    }

    /**
     * Shows the specified image on the specified swing label.
     *
     * @param img Image to be shown.
     * @param lb Swing label.
     */
    private static void showResult(Mat img, JLabel lb) {
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
            lb.setIcon(new ImageIcon(bufImage));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Draws specified boxes on the specified image.
     *
     * @param boxes Boxes.
     * @param img Image.
     */
    private static void draw(List<float[]> boxes, Mat img) {
        int h = img.height();
        int w = img.width();
        for (float[] box : boxes)
            Imgproc.rectangle(
                img,
                new Point(box[1] * w, box[0] * h),
                new Point(box[3] * w, box[2] * h),
                new Scalar(0, 255, 0)
            );
    }

    /**
     * Accepts a TensorFlow session with model and image and calculates coordinates of the boxes with faces.
     *
     * @param ses TensorFlow session.
     * @param srcImg Image with face.
     * @return Coordinates of the boxes with faces.
     */
    private static List<float[]> findFace(Session ses, Mat srcImg) {

        Mat img = new Mat();
        Imgproc.cvtColor(srcImg, img, Imgproc.COLOR_BGR2RGB);

        int h = img.height();
        int w = img.width();

        byte[][][][] pixels = new byte[1][h][w][3];

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++)
                img.get(i, j, pixels[0][i][j]);
        }

        Tensor image = Tensor.create(pixels, UInt8.class);

        List<Tensor<?>> tensors = ses.runner()
            .feed("image_tensor:0", image)
            .fetch("detection_boxes:0")
            .fetch("detection_scores:0")
            .run();

        Tensor<?> boxes = tensors.get(0);
        Tensor<?> scores = tensors.get(1);

        float[][][] boxesArr = new float[1][100][4];
        boxesArr = boxes.copyTo(boxesArr);

        float[][] scoresArr = new float[1][100];
        scoresArr = scores.copyTo(scoresArr);

        List<float[]> res = new ArrayList<>();

        for (int i = 0; i < scoresArr[0].length; i++) {
            if (scoresArr[0][i] > 0.5)
                res.add(boxesArr[0][i]);
        }

        return res;
    }

    /**
     * Creates TensorFlow graph and correspondent session based on the specified model.
     *
     * @param pathToModel Path to TensorFlow model.
     * @return TensorFlow session.
     * @throws IOException If model cannot be read.
     */
    private static Session createSession(String pathToModel) throws IOException {
        byte[] serialized = FileUtils.readFileToByteArray(new File(pathToModel));
        Graph graph = new Graph();
        graph.importGraphDef(serialized);

        return new Session(graph);
    }
}
