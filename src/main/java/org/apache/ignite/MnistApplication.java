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

import java.io.DataInputStream;
import java.io.IOException;
import java.nio.LongBuffer;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * Example of TensorFlow inference on MNIST model and data.
 */
public class MnistApplication {
    /**
     * Reads the prepared TensorFlow model for MNIST images classifications and runs it on the MNIST images.
     *
     * @param args Arguments.
     * @throws Exception In case of exception.
     */
    public static void main(String... args) throws Exception {
        SavedModelBundle smb = SavedModelBundle.load("src/main/resources/models/mnist/", "serve");

        try (Session ses = smb.session()) {
            for (MnistImage image : read()) {
                long exp = image.lb;
                long predicted = predict(ses, image.data);
                System.out.println("Expected: " + exp + ", predicted: " + predicted);
            }
        }
    }

    /**
     * Makes a prediction for the given data using specified TensorFlow session.
     *
     * @param sess TensorFlow session.
     * @param data Image represented as 3-d array.
     * @return Predicted digit.
     */
    private static long predict(Session sess, float[][][] data) {
        Tensor<?> res = sess.runner()
            .feed("Placeholder", Tensor.create(data))
            .fetch("ArgMax")
            .run()
            .iterator()
            .next();

        LongBuffer lb = LongBuffer.allocate(1);
        res.writeTo(lb);
        return lb.array()[0];
    }

    /**
     * Read MNIST images.
     *
     * @return Array of MNIST images.
     * @throws IOException If files with MNIST data can't be read.
     */
    private static MnistImage[] read() throws IOException {
        DataInputStream images = new DataInputStream(MnistApplication.class.getClassLoader()
            .getResourceAsStream("data/mnist/train-images-idx3-ubyte"));
        DataInputStream labels = new DataInputStream(MnistApplication.class.getClassLoader()
            .getResourceAsStream("data/mnist/train-labels-idx1-ubyte"));

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
            float[][][] data = new float[1][28][28];
            for (int j = 0; j < imagesRows * imagesCols; j++) {
                float px = (float)(1.0 * (images.readByte() & 0xFF) / 255);
                data[0][j / imagesCols][j % imagesCols] = px;
            }
            res[i] = new MnistImage(data, labels.readByte());
        }

        return res;
    }

    /**
     * MNIST image containing an image as a 3-d array and the digit that is shown on the image.
     */
    static class MnistImage {

        /** Image as a 3-d array. */
        final float[][][] data;

        /** Digit that is shown on the image. */
        final int lb;

        /**
         * Constructs a new instance of MNIST image.
         *
         * @param data Image as a 3-d array.
         * @param lb Digit that is shown on the image.
         */
        MnistImage(float[][][] data, int lb) {
            this.data = data;
            this.lb = lb;
        }
    }
}
