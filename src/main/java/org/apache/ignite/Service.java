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

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.opencv.core.Core;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

public class Service {

    static {
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String... args) throws IOException, ClassNotFoundException {
        byte[] serialized = FileUtils.readFileToByteArray(new File("/home/gridgain/tensorflow-inference-on-apache-ignite/src/main/resources/faces/saved_model.pb"));
        Graph graph = new Graph();
        graph.importGraphDef(serialized);

        Session session = new Session(graph);

        ServerSocket ss = new ServerSocket(8765);
        while (true) {
            try (Socket s = ss.accept()) {
                ObjectInputStream ois = new ObjectInputStream(s.getInputStream());
                ObjectOutputStream oos = new ObjectOutputStream(s.getOutputStream());

                oos.writeObject(makeMagic(session, (byte[][][][])ois.readObject()));
                oos.flush();
            }
        }
    }

    private static float[] makeMagic(Session ses, byte[][][][] pixels) {
        Tensor image = Tensor.create(pixels, UInt8.class);

        List<Tensor<?>> result = ses.runner()
            .feed("image_tensor:0", image)
            .fetch("detection_boxes:0")
            .fetch("detection_scores:0")
            .run();

        Tensor<?> boxes = result.get(0);
        Tensor<?> scores = result.get(1);

        float[][][] boxesArr = new float[1][100][4];
        boxesArr = boxes.copyTo(boxesArr);
        float[][] scoresArr = new float[1][100];
        scoresArr = scores.copyTo(scoresArr);

        for (int i = 0; i < scoresArr[0].length; i++) {
            if (scoresArr[0][i] > 0.5)
                return boxesArr[0][i];
        }

        return new float[4];
    }
}
