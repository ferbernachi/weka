import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ArffLoader;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.SerializationHelper;

import java.io.File;

/**
 * ============================================================
 * ARIKETA 3: FilteredClassifier - Modu Trinkoa
 * ============================================================
 *
 * weka.classifiers.meta.FilteredClassifier erabiltzen du ataza guztiak
 * era trinko batean egiteko: filtroa eta sailkatzailea bakar batean bilduta.
 *
 * ABANTAILAK:
 *   - Kode sinpleagoa kasu errazetan.
 *   - FilteredClassifier-ek transformazioa eta sailkapena batera kudeatzen ditu.
 *   - .model bakar bat gorde eta kargatzearekin nahikoa.
 *
 * DESABANTAILAK (zergatik ez den beti egokiena):
 *   - Malgutasun gutxiago proiektu handietarako.
 *   - Train/Test bateragarritasunaren kontrola zailagoa da.
 *   - Ariketa 1+2 konbinazio esplizitoagoa eta kontrolagarriagoa da.
 *
 * Erabilera:
 *   java Ariketa3_FilteredClassifier <train_raw.arff> <test_raw.arff> <fc_modelo.model>
 *
 * Adibidea:
 *   java Ariketa3_FilteredClassifier data/train_raw.arff data/test_raw.arff models/fc_modelo.model
 * ============================================================
 */
public class Ariketa3_FilteredClassifier {

    public static void main(String[] args) throws Exception {

        // ----------------------------------------------------------
        // 0. Argumentuak egiaztatu
        // ----------------------------------------------------------
        if (args.length < 3) {
            System.err.println("Erabilera: java Ariketa3_FilteredClassifier "
                    + "<train_raw.arff> <test_raw.arff> <fc_modelo.model>");
            System.exit(1);
        }

        String trainRawPath = args[0]; // Sarrera: train_raw.arff
        String testRawPath  = args[1]; // Sarrera: test_raw.arff
        String modelPath    = args[2]; // Irteera: fc_modelo.model

        // ----------------------------------------------------------
        // 1. Datuak kargatu
        // ----------------------------------------------------------
        System.out.println(">>> 1. URRATSA: Datuak kargatu");

        ArffLoader loader = new ArffLoader();

        loader.setFile(new File(trainRawPath));
        Instances trainRaw = loader.getDataSet();
        trainRaw.setClassIndex(trainRaw.numAttributes() - 1);
        System.out.println("    Train instantziak : " + trainRaw.numInstances());

        loader.setFile(new File(testRawPath));
        Instances testRaw = loader.getDataSet();
        testRaw.setClassIndex(testRaw.numAttributes() - 1);
        System.out.println("    Test instantziak  : " + testRaw.numInstances());

        // ----------------------------------------------------------
        // 2. StringToWordVector filtroa konfiguratu
        // ----------------------------------------------------------
        System.out.println("\n>>> 2. URRATSA: StringToWordVector konfiguratu");

        StringToWordVector filter = new StringToWordVector();
        filter.setOutputWordCounts(true); // BoW kopuruak
        filter.setWordsToKeep(10000);     // Hitz kopuru maximoa hiztegian
        filter.setLowerCaseTokens(true);  // Letra xehera
        filter.setTFTransform(true);      // TF eraldaketa
        filter.setIDFTransform(true);     // IDF eraldaketa

        // ----------------------------------------------------------
        // 3. Oinarrizko sailkatzailea (NaiveBayes)
        // ----------------------------------------------------------
        Classifier baseSailkatzailea = new NaiveBayes();

        // ----------------------------------------------------------
        // 4. FilteredClassifier sortu: filtroa + sailkatzailea batera
        // ----------------------------------------------------------
        System.out.println("\n>>> 3. URRATSA: FilteredClassifier sortu eta entrenatu");

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(filter);                 // BoW filtroa ezarri
        fc.setClassifier(baseSailkatzailea);  // NaiveBayes ezarri

        // Entrenamendua: FilteredClassifier-ek barne-barne egiten du:
        //   1) filtroa train datuekin doitu (hiztegia sortu)
        //   2) train datuak transformatu
        //   3) sailkatzailea transformatutako datuekin entrenatu
        fc.buildClassifier(trainRaw);
        System.out.println("    FilteredClassifier entrenatuta.");

        // ----------------------------------------------------------
        // 5. FilteredClassifier gorde
        // ----------------------------------------------------------
        System.out.println("\n>>> 4. URRATSA: FilteredClassifier gorde  [" + modelPath + "]");
        SerializationHelper.write(modelPath, fc);
        System.out.println("    Gordeta: " + modelPath);

        // ----------------------------------------------------------
        // 6. Test multzoko iragarpenak egin
        // ----------------------------------------------------------
        System.out.println("\n>>> 5. URRATSA: Test iragarpenak egin");
        System.out.println("    -----------------------------------------------------");
        System.out.printf("    %-6s  %-20s  %-20s%n", "Indize", "Erreala", "Iragarpena");
        System.out.println("    -----------------------------------------------------");

        int zuzenak = 0;
        for (int i = 0; i < testRaw.numInstances(); i++) {
            Instance instance = testRaw.instance(i);

            String erreala = instance.classIsMissing()
                    ? "?"
                    : testRaw.classAttribute().value((int) instance.classValue());

            // FilteredClassifier-ek barne-barne egiten du:
            //   1) instantzia transformatu (train hiztegiarekin bat etorrita)
            //   2) sailkatu transformatutako instantzia
            double iragarpenIndizea = fc.classifyInstance(instance);
            String iragarpena = testRaw.classAttribute().value((int) iragarpenIndizea);

            System.out.printf("    %-6d  %-20s  %-20s%n", (i + 1), erreala, iragarpena);

            if (!instance.classIsMissing() && erreala.equals(iragarpena)) {
                zuzenak++;
            }
        }

        // ----------------------------------------------------------
        // LABURPENA
        // ----------------------------------------------------------
        System.out.println("    -----------------------------------------------------");
        System.out.println("\n========== ARIKETA 3 AMAITUTA ==========");
        System.out.println("FilteredClassifier modeloa gordeta: " + modelPath);
        if (!testRaw.instance(0).classIsMissing()) {
            double zehaztasuna = 100.0 * zuzenak / testRaw.numInstances();
            System.out.printf("Zehaztasuna (Accuracy) : %d / %d  (%.2f%%)%n",
                    zuzenak, testRaw.numInstances(), zehaztasuna);
        }

        System.out.println("\n--- KONPARAKETA: Ariketa 1+2 vs Ariketa 3 ---");
        System.out.println("  Ariketa 1+2 (bereizia)  : malgua, train/test esplizitoagoa, proiektu handietarako");
        System.out.println("  Ariketa 3 (FilteredClassifier): trinkoa, ataza sinpleetarako egokia");
    }
}
