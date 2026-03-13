import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

import java.io.File;

/**
 * ============================================================
 * ARIKETA 2: Iragarpenak Egin
 * ============================================================
 *
 * Egiten duena (bi urrats):
 *   URRATS 1) test_raw.arff  -[FilteredStringToWordVector]->  test_BoW.arff
 *              KONTUZ: train-eko HIZTEGI BERA erabili behar da bateragarritasuna
 *              bermatzeko. Horretarako Ariketa1-ean gordetako filtroa kargatzen da.
 *
 *   URRATS 2) test_BoW.arff  -[sailkatzailea.model]-------->  iragarpenak
 *
 * Erabilera:
 *   java Ariketa2_IragarpenakEgin <test_raw.arff> <test_BoW.arff>
 *                                 <sailkatzailea.model> <sailkatzailea_filter.model>
 *
 * Adibidea:
 *   java Ariketa2_IragarpenakEgin data/test_raw.arff data/test_BoW.arff
 *                                 models/sailkatzailea.model models/sailkatzailea_filter.model
 * ============================================================
 */
public class Ariketa2_IragarpenakEgin {

    public static void main(String[] args) throws Exception {

        // ----------------------------------------------------------
        // 0. Argumentuak egiaztatu
        // ----------------------------------------------------------
        if (args.length < 4) {
            System.err.println("Erabilera: java Ariketa2_IragarpenakEgin "
                    + "<test_raw.arff> <test_BoW.arff> "
                    + "<sailkatzailea.model> <sailkatzailea_filter.model>");
            System.exit(1);
        }

        String testRawPath  = args[0]; // Sarrera:  test_raw.arff
        String testBoWPath  = args[1]; // Irteera:  test_BoW.arff
        String modelPath    = args[2]; // Sarrera:  sailkatzailea.model   (Ariketa1-etik)
        String filterPath   = args[3]; // Sarrera:  sailkatzailea_filter.model (Ariketa1-etik)

        // ----------------------------------------------------------
        // 1. test_raw.arff kargatu
        // ----------------------------------------------------------
        System.out.println(">>> 1. URRATSA: Test datuak kargatu  [" + testRawPath + "]");
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(testRawPath));
        Instances testRaw = loader.getDataSet();
        testRaw.setClassIndex(testRaw.numAttributes() - 1);
        System.out.println("    Instantziak : " + testRaw.numInstances());
        System.out.println("    Atributuak  : " + testRaw.numAttributes());

        // ----------------------------------------------------------
        // 2. Ariketa1-eko filtroa kargatu (train hiztegiarekin)
        // ----------------------------------------------------------
        System.out.println("\n>>> 2. URRATSA: Train filtroa kargatu  [" + filterPath + "]");
        // Filtro honek entrenamenduan eraikitako hiztegia dauka barru-barruan.
        // Horrela test multzoko testu berriak ATRIBUTU ESPAZIO BERDINERA proiektatzen dira.
        StringToWordVector filter = (StringToWordVector) SerializationHelper.read(filterPath);
        System.out.println("    Filtroa kargatuta (train hiztegiarekin).");

        // ----------------------------------------------------------
        // 3. Filtroa test multzoan aplikatu (train atributu espaziora egokituz)
        // ----------------------------------------------------------
        System.out.println("\n>>> 3. URRATSA: Test multzoa BoW-era transformatu (train espaziora)");

        // GILTZARRIA: setInputFormat TEST datuekin deitzen da BAINA filtroak
        // dagoeneko train hiztegia dauka --> ez da hiztegi berririk sortzen.
        // Horrela test-eko atributu-espazioa train-ekoarekin BATERAGARRIA da.
        filter.setInputFormat(testRaw);
        Instances testBoW = Filter.useFilter(testRaw, filter);
        testBoW.setClassIndex(testBoW.numAttributes() - 1);

        System.out.println("    BoW ondorengo atributuak : " + testBoW.numAttributes());
        System.out.println("    Instantziak              : " + testBoW.numInstances());

        // ----------------------------------------------------------
        // 4. test_BoW.arff gorde
        // ----------------------------------------------------------
        System.out.println("\n>>> 4. URRATSA: test_BoW.arff gorde  [" + testBoWPath + "]");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(testBoW);
        saver.setFile(new File(testBoWPath));
        saver.writeBatch();
        System.out.println("    Gordeta: " + testBoWPath);

        // ----------------------------------------------------------
        // 5. Sailkatzailea kargatu (Ariketa1-etik)
        // ----------------------------------------------------------
        System.out.println("\n>>> 5. URRATSA: Sailkatzailea kargatu  [" + modelPath + "]");
        Classifier sailkatzailea = (Classifier) SerializationHelper.read(modelPath);
        System.out.println("    Sailkatzailea kargatuta: " + sailkatzailea.getClass().getSimpleName());

        // ----------------------------------------------------------
        // 6. Test multzoko iragarpenak egin
        // ----------------------------------------------------------
        System.out.println("\n>>> 6. URRATSA: Iragarpenak egin");
        System.out.println("    -----------------------------------------------");
        System.out.printf("    %-6s  %-20s  %-20s%n", "Indize", "Erreala", "Iragarpena");
        System.out.println("    -----------------------------------------------");

        int zuzenak = 0;
        for (int i = 0; i < testBoW.numInstances(); i++) {
            Instance instance = testBoW.instance(i);

            // Klase erreala (etiketatuta badago)
            String erreala = instance.classIsMissing()
                    ? "?"
                    : testBoW.classAttribute().value((int) instance.classValue());

            // Iragarpenaren indizea --> klase izena lortu
            double iragarpenIndizea = sailkatzailea.classifyInstance(instance);
            String iragarpena = testBoW.classAttribute().value((int) iragarpenIndizea);

            System.out.printf("    %-6d  %-20s  %-20s%n", (i + 1), erreala, iragarpena);

            if (!instance.classIsMissing() && erreala.equals(iragarpena)) {
                zuzenak++;
            }
        }

        // ----------------------------------------------------------
        // LABURPENA
        // ----------------------------------------------------------
        System.out.println("    -----------------------------------------------");
        System.out.println("\n========== ARIKETA 2 AMAITUTA ==========");
        System.out.println("Sortutako fitxategiak:");
        System.out.println("  [1] " + testBoWPath + "  <-- test datuak BoW/TF-IDF formatuan");
        if (!testBoW.instance(0).classIsMissing()) {
            double zehaztasuna = 100.0 * zuzenak / testBoW.numInstances();
            System.out.printf("Zehaztasuna (Accuracy) : %d / %d  (%.2f%%)%n",
                    zuzenak, testBoW.numInstances(), zehaztasuna);
        }
    }
}
