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

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.PixelGrabber;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.apache.commons.io.FileUtils;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.types.UInt8;

public class Application {

    public static void main(String... args) throws Exception {
        System.out.println(TensorFlow.version());

        byte[] serialized = FileUtils.readFileToByteArray(new File("/home/gridgain/tensorflow-inference-on-apache-ignite/src/main/resources/faces/saved_model.pb"));
//        byte[] img = FileUtils.readFileToByteArray(new File("/home/gridgain/tensorflow-inference-on-apache-ignite/src/main/resources/1506756159.jpg"));

        BufferedImage img = ImageIO.read(new File("/home/gridgain/tensorflow-inference-on-apache-ignite/src/main/resources/1506756159.jpg"));

        int h = img.getHeight();
        int w = img.getWidth();

        int[] pixels = new int[w * h];
        PixelGrabber pg = new PixelGrabber(img, 0, 0, w, h, pixels, 0, w);
        pg.grabPixels();

        int[][][][] data = new int[1][w][h][3];
        for (int i = 0; i < pixels.length; i++) {
            String temp = Integer.toString(pixels[i]);
            data[0][i / h][i % h][0] = Color.decode(temp).getRed();
            data[0][i / h][i % h][1] = Color.decode(temp).getBlue();
            data[0][i / h][i % h][2] = Color.decode(temp).getGreen();
            System.out.println(Arrays.toString(data[0][i / h][i % h]));
        }

        Graph graph = new Graph();
        graph.importGraphDef(serialized);

        Session session = new Session(graph);

        Tensor image = Tensor.create(data, UInt8.class);
        System.out.println("Image type: " + image.dataType());
        List<Tensor<?>> result = session.runner()
            .feed("image_tensor:0",image)
            .fetch("detection_boxes:0")
            .fetch("detection_scores:0")
            .fetch("detection_classes:0")
            .fetch("num_detections:0")
            .run();

        Tensor<?> boxes = result.get(0);
        Tensor<?> scores = result.get(1);
        Tensor<?> classes = result.get(2);
        Tensor<?> detections = result.get(3);

        float[][][] arr = new float[1][100][4];
        arr = boxes.copyTo(arr);
        for (int i = 0; i < 100; i++) {
            float[] box = arr[0][i];
            System.out.println("Box: " + Arrays.toString(box));
        }

        float[] yyy = new float[1];
        yyy = detections.copyTo(yyy);

        System.out.println("Detections: " + Arrays.toString(yyy));

//        try (SavedModelBundle bundle = SavedModelBundle.load("/home/gridgain/tensorflow-inference-on-apache-ignite/src/main/resources/faces/")) {
////        try (SavedModelBundle bundle = SavedModelBundle.load("/home/gridgain/model/1541682474/", "serve")) {
//            Operation op = bundle.graph().operation("image_tensor:0");
//            System.out.println("Op: " + op);
////            for (MnistImage image : read()) {
////                long expRes = image.lb;
////                long predictedRes = predict(bundle.session(), image.pixels, image.data);
////                System.out.println("Expected: " + expRes + ", predicted: " + predictedRes);
////            }
//        }
    }

    private static void printOperation(Graph graph) {
        Iterator<Operation> operations = graph.operations();
        while (operations.hasNext()) {
            Operation operation = operations.next();
            System.out.println(operation.name() + " (" + operation.type() + ")");
        }
        System.out.println("-----------------------------------------------");
    }

    private static void printSignature(SavedModelBundle model) throws Exception {
        MetaGraphDef m = MetaGraphDef.parseFrom(model.metaGraphDef());
        SignatureDef sig = m.getSignatureDefOrThrow("serving_default");
        int numInputs = sig.getInputsCount();
        int i = 1;
        System.out.println("MODEL SIGNATURE");
        System.out.println("Inputs:");
        for (Map.Entry<String, TensorInfo> entry : sig.getInputsMap().entrySet()) {
            TensorInfo t = entry.getValue();
            System.out.printf(
                "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
                i++, numInputs, entry.getKey(), t.getName(), t.getDtype());
        }
        int numOutputs = sig.getOutputsCount();
        i = 1;
        System.out.println("Outputs:");
        for (Map.Entry<String, TensorInfo> entry : sig.getOutputsMap().entrySet()) {
            TensorInfo t = entry.getValue();
            System.out.printf(
                "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
                i++, numOutputs, entry.getKey(), t.getName(), t.getDtype());
        }
        System.out.println("-----------------------------------------------");
    }

    private static long predict(Session sess, FloatBuffer buf, float[][][] data) {
        List<Tensor<?>> res = sess.runner()
            .feed("Placeholder", Tensor.create(data))
            .fetch("ArgMax")
            .run();

        for (Tensor<?> t : res) {
            LongBuffer lb = LongBuffer.allocate(1);
            t.writeTo(lb);
            return lb.array()[0];
        }

        return -1;
    }

    private static MnistImage[] read() throws IOException {
        DataInputStream images = new DataInputStream(Application.class.getClassLoader()
            .getResourceAsStream("mnist/train-images-idx3-ubyte"));
        DataInputStream labels = new DataInputStream(Application.class.getClassLoader()
            .getResourceAsStream("mnist/train-labels-idx1-ubyte"));

        images.readInt();
        int imagesLen = images.readInt();
        int imagesRows = images.readInt();
        int imagesCols = images.readInt();

        labels.readInt();
        int labelsLen = labels.readInt();

        if (imagesLen != labelsLen)
            throw new IllegalStateException("Number of images and labels are not equal");

        MnistImage[] res = new MnistImage[imagesLen];

        for (int i = 0; i < imagesLen; i++) {
            FloatBuffer pixels = FloatBuffer.allocate(imagesRows * imagesCols);
            float[][][] data = new float[1][28][28];
            for (int j = 0; j < imagesRows * imagesCols; j++) {
                float px = (float)(1.0 * (images.readByte() & 0xFF) / 255);
                pixels.put(px);
                data[0][j / imagesCols][j % imagesCols] = px;
            }
            res[i] = new MnistImage(pixels, data, labels.readByte());
        }

        return res;
    }

    static class MnistImage {

        final FloatBuffer pixels;

        final float[][][] data;

        final int lb;

        MnistImage(FloatBuffer pixels, float[][][] data, int lb) {
            this.pixels = pixels;
            this.data = data;
            this.lb = lb;
        }
    }
}
