import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.SerializationHelper;

import java.io.File;

/**
 * ============================================================
 * ARIKETA 1: Sailkatzailea Sortu
 * ============================================================
 *
 * Egiten duena (bi urrats):
 *   URRATS 1) train_raw.arff  -[StringToWordVector]->  train_BoW.arff
 *   URRATS 2) train_BoW.arff  -[NaiveBayes]--------->  sailkatzailea.model
 *
 * GARRANTZITSUA:
 *   StringToWordVector filtroa ere gordetzen da (sailkatzailea_filter.model)
 *   Ariketa2-n test multzoa bateragarri bihurtzeko erabiliko dena.
 *   Hiztegia train batch-etik sortzen da (setInputFormat deitzen denean).
 *
 * Erabilera:
 *   java Ariketa1_SailkatzaileSortu <train_raw.arff> <train_BoW.arff> <sailkatzailea.model>
 *
 * Adibidea:
 *   java Ariketa1_SailkatzaileSortu data/train_raw.arff data/train_BoW.arff models/sailkatzailea.model
 * ============================================================
 */
public class Ariketa1_SailkatzaileSortu {

    public static void main(String[] args) throws Exception {

        // ----------------------------------------------------------
        // 0. Argumentuak egiaztatu
        // ----------------------------------------------------------
        if (args.length < 3) {
            System.err.println("Erabilera: java Ariketa1_SailkatzaileSortu "
                    + "<train_raw.arff> <train_BoW.arff> <sailkatzailea.model>");
            System.exit(1);
        }

        String trainRawPath = args[0]; // Sarrera:  train_raw.arff
        String trainBoWPath = args[1]; // Irteera:  train_BoW.arff
        String modelPath    = args[2]; // Irteera:  sailkatzailea.model
        // Filtroa automatikoki gordetzen da izenaz: sailkatzailea_filter.model
        String filterPath   = modelPath.replace(".model", "_filter.model");

        // ----------------------------------------------------------
        // 1. train_raw.arff kargatu
        // ----------------------------------------------------------
        System.out.println(">>> 1. URRATSA: Train datuak kargatu  [" + trainRawPath + "]");
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(trainRawPath));
        Instances trainRaw = loader.getDataSet();
        // Klase atributua ezarri (azken atributua)
        trainRaw.setClassIndex(trainRaw.numAttributes() - 1);
        System.out.println("    Instantziak : " + trainRaw.numInstances());
        System.out.println("    Atributuak  : " + trainRaw.numAttributes());

        // ----------------------------------------------------------
        // 2. StringToWordVector filtroa konfiguratu eta aplikatu
        // ----------------------------------------------------------
        System.out.println("\n>>> 2. URRATSA: StringToWordVector konfiguratu eta aplikatu");

        StringToWordVector filter = new StringToWordVector();

        // [outputWordCounts]
        //   true  --> hitz bakoitzaren kopurua irteera gisa (BoW kopuruak)
        //   false --> presentzia bitarra (0 edo 1)
        filter.setOutputWordCounts(true);

        // [wordsToKeep] Hiztegian gordeko diren hitzen kopuru maximoa
        filter.setWordsToKeep(10000);

        // [lowercase] Testu guztia letra xehera bihurtu konparaketak egiteko
        filter.setLowerCaseTokens(true);

        // TF-IDF errepresentazioa (bi flag hauek aktibatuta lortzen da):
        filter.setTFTransform(true);  // TF: Term Frequency normalizazioa
        filter.setIDFTransform(true); // IDF: Inverse Document Frequency pisua

        // setInputFormat --> HIZTEGIA SORTU: lehenengo batch honetan (train)
        // zehaztuko da ze hitzek osatuko duten atributu-espazioa
        filter.setInputFormat(trainRaw);

        // Filtroa train multzoan aplikatu (hiztegia train-etik eratorriko da)
        Instances trainBoW = Filter.useFilter(trainRaw, filter);
        trainBoW.setClassIndex(trainBoW.numAttributes() - 1);

        System.out.println("    BoW ondorengo atributuak : " + trainBoW.numAttributes());
        System.out.println("    Instantziak              : " + trainBoW.numInstances());

        // ----------------------------------------------------------
        // 3. train_BoW.arff fitxategian gorde
        // ----------------------------------------------------------
        System.out.println("\n>>> 3. URRATSA: train_BoW.arff gorde  [" + trainBoWPath + "]");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(trainBoW);
        saver.setFile(new File(trainBoWPath));
        saver.writeBatch();
        System.out.println("    Gordeta: " + trainBoWPath);

        // ----------------------------------------------------------
        // 4. Sailkatzailea entrenatu (NaiveBayes)
        // ----------------------------------------------------------
        System.out.println("\n>>> 4. URRATSA: Sailkatzailea entrenatu (NaiveBayes)");
        Classifier sailkatzailea = new NaiveBayes();
        sailkatzailea.buildClassifier(trainBoW);
        System.out.println("    Entrenamendua amaituta.");

        // ----------------------------------------------------------
        // 5. Modeloa eta filtroa gorde serializations bidez
        // ----------------------------------------------------------
        System.out.println("\n>>> 5. URRATSA: Modeloak gorde");

        // Sailkatzailea gorde
        SerializationHelper.write(modelPath, sailkatzailea);
        System.out.println("    Sailkatzailea gordeta : " + modelPath);

        // Filtroa gorde --> Ariketa2-n test multzoa hiztegiarekin bateragarri bihurtzeko
        // Filtro honek entrenamenduko hiztegia barruan dauka
        SerializationHelper.write(filterPath, filter);
        System.out.println("    Filtroa gordeta       : " + filterPath);

        // ----------------------------------------------------------
        // LABURPENA
        // ----------------------------------------------------------
        System.out.println("\n========== ARIKETA 1 AMAITUTA ==========");
        System.out.println("Sortutako fitxategiak:");
        System.out.println("  [1] " + trainBoWPath + "  <-- train datuak BoW/TF-IDF formatuan");
        System.out.println("  [2] " + modelPath    + "  <-- entrenatutako sailkatzailea");
        System.out.println("  [3] " + filterPath   + "  <-- filtroa (Ariketa2-n erabiltzeko)");
    }
}
