/**
 * Alunos: Silvana Trindade e Maurício André Cinelli
 * Trabalho V: Classificação de objetos
 * Opencv versão 3.0
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <random>
#include <stack>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace cv;

double pi = 3.1415926535897;
const int DELAY_AQUISICAO = 10;

class Modelo
{
public:
  MatND histograma;
  float area;
  float solidity;
  float equi_diametro;
  int countPixels;

  float regraDeTres(float a, float  b)
  {
    float res = 0.0f;

    if (b > a)
    {
      res = ((100.0f * a) / b);
    }
    else
    {
      res = ((100.0f * b) / a);
    }

    res /= 100.0f;

    return res;
  }

  float testaArea(float a)
  {
    return regraDeTres(area, a);
  }

  float testaCountPixels(int c)
  {
    return regraDeTres((float)countPixels, (float)c);
  }

  float testaHistograma(MatND h)
  {
    float res_histograma = compareHist(h, histograma, CV_COMP_CORREL);
    return res_histograma;
  }
};

vector<Modelo> modelos;
Scalar cores[] = {
  Scalar(255, 0, 0),//1
  Scalar(255, 255, 0),//2
  Scalar(255, 0, 255),//3
  Scalar(0, 255, 0),//4
  Scalar(0, 0, 255),//5
  Scalar(0, 0, 128),//6
  Scalar(0, 128, 128),//7
  Scalar(128, 0, 255),//8
  Scalar(0,250, 255),//9
  Scalar(0, 255, 255),//10
  Scalar(255, 182, 0),//11
  Scalar(0,0,0),//12
  Scalar(255,255,255),//13
  Scalar(59, 255, 0),//14
  Scalar(127, 0, 255),//15
  Scalar(255, 242, 0),//16
  Scalar(147, 103, 103),//17
  Scalar(0, 255, 33),//18
  Scalar(163, 70, 153),//19
  Scalar(0, 140, 255),//20
  Scalar(158, 170, 102),//21
  Scalar(255, 84, 0),//22
  Scalar(53, 84, 50),//23
  Scalar(72, 79, 145),//24
  Scalar(90, 27, 163),//25
  Scalar(12, 55, 140),//26
  Scalar(137, 26, 26),//27
  Scalar(201, 98, 98),//28
  Scalar(216, 52, 131),//29
  Scalar(244, 204, 0)//30


};

float distanciaEuclidiana(Point& p, Point& q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

class Objeto {
public:

  Mat regiao;
  Mat frame;
  Mat fbproj;
  Mat fhsv;
  Mat fbthr;
  Rect wind;
  MatND rhist;
  Mat mask;
  Mat rhsv;
  int classe;
  vector<Point> contorno;

  Objeto(Mat imagem, Rect r, vector<Point> c)
  {
    // converte imagem para HSV para histograma
    cvtColor(imagem, rhsv, CV_BGR2HSV);

    contorno = c;
    Rect bb = boundingRect(c);

    // area do contorno
    float area = contourArea(c);

    // Calcula o convex hull do objeto
    // que é um poligono que envolve o objeto
    vector< Point > _hull;
    convexHull( Mat(contorno), _hull, false );

    // Área do convex hull
    float _hull_area = contourArea(_hull);

    // Solidity = o quão sólido o objeto é
    double _sol = area/_hull_area;

    // diâmetro
    double _equi_diametro = sqrt(4*area/pi);

    // contagem de pixels
    // como imagem do objeto tem fundo preto
    // basta contar numero de pixels "não-zero"
    Mat gray;
    cvtColor(imagem, gray, CV_RGB2GRAY);
    int _countPixels = countNonZero(gray);

    // calcula histograma
    int rhistsz = 180;    // bin size
    float range[] = { 0, 180};
    const float *ranges[] = { range };
    int channels[] = {0};
    calcHist( &rhsv, 1, channels, Mat(), rhist, 1, &rhistsz, ranges, true, false );
    normalize(rhist,rhist,0,255,NORM_MINMAX, -1, Mat() );

    wind = r;

    /**
     * Acha classe do objeto ou cria uma nova
     */

    int _classe = -1;
    double maior = -1;

    for (int j = 0; j < (int)modelos.size(); j++) {

      /**
       * Obtêm a similaridade dos objetos
       * testa o quão similar, em porcentagem, um objeto
       * é em relação ao model em diferentes medidas
       */
      double _area = modelos[j].testaArea(area);
      double _histograma = modelos[j].testaHistograma(rhist);
      double _count_pixels = modelos[j].testaCountPixels(_countPixels);

      // Fórmula para agregar as medidas de similaridade
      double similaridade = 0.18f*_histograma + 0.48f*_area + 0.34f*_count_pixels;

      if (similaridade > 0.70f && similaridade > maior)
      {
        _classe = j;
        maior = similaridade;
      }
    }

    // se não encontrou classe, cria uma nova
    // utilizando o objeto como modelo
    if (_classe == -1)
    {
      Modelo m;
      m.area = area;
      m.histograma = rhist;
      m.solidity = _sol;
      m.equi_diametro = _equi_diametro;
      m.countPixels = _countPixels;

      modelos.push_back(m);

      _classe = (int)modelos.size() - 1;
    }

    classe = _classe;
  }

  // A partir de uma lista de contornos tenta encontrar
  // um objeto cuja distancia entre os centros seja
  // pequena, indicando que é o mesmo objeto do frame anterior
  int track(vector< vector<Point> > &contornos, Mat &image) {
    int menor_indice = -1;
    double menor_distancia = std::numeric_limits<double>::max();
    bool achou = false;
    Point centro(wind.x + (wind.width / 2), wind.y + (wind.height / 2));

    for (int i = 0; i < (int) contornos.size(); i++)
    {
      Rect bb = boundingRect(contornos[i]);
      Point c(bb.x + (bb.width / 2), bb.y + (bb.height / 2));

      float res = distanciaEuclidiana(c, centro);

      if (res < 25 && res < menor_distancia) {
        menor_distancia = res;
        menor_indice = i;
        achou = true;
      }
    }

    if (!achou) {
      return -1;
    }

    wind = boundingRect(contornos[menor_indice]);

    rectangle(image, wind, cores[classe], 1);

    return 1;
  }
};

Mat img, imgtemp;
Rect roi;
Point p1, p2, p3;
bool flag1 = false, flag2 = false, flag3 = false, flag4 = true;

void on_mouse( int event, int x, int y, int flags, void* param )
{

  if (!flag3)
    return;

  switch(event)
  {
    case CV_EVENT_LBUTTONDOWN:
        if (!flag2) {
          p3.x = x;
          p3.y = y;
          flag1 = true;
        }
        break;
    case CV_EVENT_MOUSEMOVE:
        imgtemp = img.clone();
        if (flag1) {
                    rectangle(imgtemp,p3, Point(x,y+5),  CV_RGB(255,255,255),1);
        }
        imshow( "Imagem", imgtemp);
        break;
    case CV_EVENT_LBUTTONUP:
        p1.x = p3.x;
        p1.y = p3.y;
        p2.x = x;
        p2.y = y;
        flag1 = false;
        flag2 = true;
        break;
    default:
      break;
  }
}

bool pegaROI() {

  imgtemp = img.clone();

  if(img.empty())
     return -1;

  flag1 = flag2 = false;
  flag3 = true;

  p1.x = p1.y = 10;
  p2.x = img.cols-10;
  p2.y = img.rows-10;


  cvNamedWindow( "Imagem", CV_WINDOW_AUTOSIZE );
  setMouseCallback("Imagem", on_mouse, 0 );
  imshow( "Imagem", img);

  if (img.rows< 50 || img.cols < 50) {
      printf("Erro: imagem ḿuito pequena!\n");
      return -1;
  }
  imgtemp = img.clone();

  while(!flag2) {
    if(waitKey(10) == 27) {
      break;
    }
    if (flag2) {
      break;
    }
  }

  cvDestroyWindow("Imagem");
    roi = Rect(p1,p2);
    return true;
}

int main(int argc, char *argv[]) {
  // verifica se vai pegar video de arquivo ou câmera
  bool stream = false;

  if (argc < 2)
  {
      stream = true;
  }

  VideoCapture *cap;

  if (stream) {
    cap = new VideoCapture(1);
  } else {
    cap = new VideoCapture(argv[1]);
  }

  if(!cap->isOpened())
      return -1;

  Mat frame;

  // Cria subtração de background por MOG2
  Ptr< BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(500,60,true);
  Mat mascara_background;
  Mat binaryImg;

  mog2->setBackgroundRatio(0.01);

  for (int i = 0; i < 80; i++)
  {
    *cap >> frame;
    mog2->apply(frame, mascara_background);
  }

  img = frame.clone();
  namedWindow("video", WINDOW_AUTOSIZE);

  // matriz de morfologia
  Mat elemento = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1,1) );

  pegaROI();

  // vetor de contadores independentes, por classe
  vector<int> contadores(30, 0);

  vector<Objeto> objetos;
  bool pausar = false;

  // largura e altura do frame
  double hframe = frame.size().height;
  double wframe = frame.size().width;

  // pontos para linha superior
  Point p1(0, roi.y);
  Point p2(wframe, roi.y);

  // pontos para linha inferior
  Point p3(0, roi.y + roi.height + 5);
  Point p4(wframe, roi.y + roi.height + 5);

  while (true) {
    if (pausar == false)
    {
      *cap >> frame;
      if(frame.empty()) {
        break;
      }

      // aplica frame ao background
      mog2->apply(frame, mascara_background);

      // remove sombras
      threshold(mascara_background, binaryImg,130, 255, CV_THRESH_BINARY);

      Mat binOrig = binaryImg.clone();

      morphologyEx(binaryImg, binaryImg, CV_MOP_ERODE, elemento);
      morphologyEx(binaryImg, binaryImg, CV_MOP_ERODE, elemento);
      
      for (int w = 0; w < 1; w++)
      {
        morphologyEx(binaryImg,binaryImg,CV_MOP_OPEN,elemento);
        morphologyEx(binaryImg,binaryImg,CV_MOP_CLOSE,elemento);
      }
      

      Mat ContourImg = binaryImg.clone();

      // pega contornos
      vector< vector<Point> > contornos;
      findContours(ContourImg, contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

      // Cria uma imagem preta, pintando os contornos de branco
      Mat mask = Mat::zeros(ContourImg.rows, ContourImg.cols, CV_8UC1);
      drawContours(mask, contornos, -1, Scalar(255, 255, 255), CV_FILLED);

      // cria uma nova imagem, sobrepondo o frame com a máscara
      // assim pinta somente os objetos, num fundo preto
      Mat crop(frame.rows, frame.cols, frame.type());
      crop.setTo(Scalar(0,0,0));
      frame.copyTo(crop, mask);


      for (int i = 0; i < (int)contornos.size(); i++)
      {
        Rect bb = boundingRect(contornos[i]);

        // objeto muito pequeno, cai fora
        if (bb.width <= 4 || bb.height <= 4 || bb.width >= 70 || bb.height >= 50)
        {
          continue;
        }

        // verifica se objeto já foi detectado em um frame anterior
        // por exemplo se a leitura e muito rapida
        bool achou = false;

        for (int j = 0; j < (int)objetos.size(); j++) {
          Rect intersecao = bb & objetos[j].wind;

          if (intersecao.height > 4 || intersecao.width > 4) {
            achou = true;
          }
        }

        if (achou)
        {
          continue;
        }

        // se não encontrar, cria um novo objeto
        if (bb.y >= roi.y && bb.y <= roi.y + 45) {
          Objeto o(crop(bb), bb, contornos[i]);
          objetos.push_back(o);
        }
      }

      // percorre objetos, chamando a função `track`
      // de cada objeto para pegar sua proxima posição
      vector<Objeto> novos_objetos;
      for (int i = 0; i < (int) objetos.size(); i++) {
        int result = objetos[i].track(contornos, frame);

        if (result == -1) continue;

        // se o objeto passou da parte de baixo do quadrado
        // conta o objeto e remove-o
        if (objetos[i].wind.y > roi.y + roi.height + 5) {
          contadores[objetos[i].classe] += 1;
          continue;
        }

        // se objeto deve continuar, coloca-o em um novo vetor
        novos_objetos.push_back(objetos[i]);
      }

      objetos = novos_objetos;

      // mostra os contadores
      int offset = 1;
      for (int i = 0; i < (int) contadores.size(); i++)
      {
        if (contadores[i] < 1) {
          continue;
        }

        // mostra contagem na tela
        string text = "Contador: " + to_string(contadores[i]);
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.7f;
        int thickness = 2;
        cv::Point textOrg(10, 30 * offset);

        putText(frame, text, textOrg, fontFace, fontScale, cores[i], thickness,8);
        offset++;
      }

      line(frame, p1, p2, Scalar(0, 0, 255), 3);
      line(frame, p3, p4, Scalar(0, 0, 255), 3);

      imshow("video", frame);
      // imshow("Binario", mascara_background);
    }

    int key =  waitKey(DELAY_AQUISICAO);

    if (key == 27)
    {
      break;
    }
    else if (key == 112)
    {
      pausar = !pausar;
    }
  }

  // mostra contadores no terminal também
  int total = 0;
  for (int i = 0; i < (int)contadores.size(); i++)
  {
    if (contadores[i] < 1) {
      continue;
    }

    total = contadores[i] + total;
    cout << "Contador " << i << ": " << contadores[i] << endl;
  }

  cout<<"Total = "<<total<<endl;

  return 0;
}

