/**
 * Alunos: Silvana Trindade e Maurício André Cinelli
 * Trabalho IV: tracking e contagem de objetos com CamShift
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

class Modelo
{
public:  
  MatND histograma;
  float area;
  float hull_area;
  float perimetro;
  float solidity;
  float equi_diametro;

  float testaArea(MatND h, float a)
  {
    float res_area = 0.0f;

    if (a > area)
    {
      res_area = ((100.0f * area) / a);
    }
    else
    {
      res_area = ((100.0f * a) / area);
    }

    res_area /= 100.0f;

    // cout << res_area << " " << res_histograma << endl;
    return res_area;
  }

  float testaHullArea(float h)
  {
    float res_area = 0.0f;

    if (h > hull_area)
    {
      res_area = ((100.0f * hull_area) / h);
    }
    else
    {
      res_area = ((100.0f * h) / hull_area);
    }
    
    // cout<<" res_area "<<res_area<<endl;
    res_area /= 100.0f;
    // float solidity = a/hull_area;
    // cout<<"solidity "<<solidity<<endl;

    return res_area;
  }

  float testaSolidity(float s)
  {
    float res_solidity = 0.0f;

    if (s > solidity)
    {
      res_solidity = ((100.0f * solidity) / s);
    }
    else
    {
      res_solidity = ((100.0f * s) / solidity);
    }
    
    // cout<<" res_area "<<res_area<<endl;
    res_solidity /= 100.0f;
    
    cout<<"solidity "<<solidity<<endl;

    return res_solidity;
  }

  float testaDiamentroEquivalente(float ed)
  {
    float res_eq_diametro = 0.0f;

    if (ed > equi_diametro)
    {
      res_eq_diametro = ((100.0f * equi_diametro) / ed);
    }
    else
    {
      res_eq_diametro = ((100.0f * ed) / equi_diametro);
    }
    
    // cout<<" res_area "<<res_area<<endl;
    res_eq_diametro /= 100.0f;
    
    cout<<"equi_diametro "<<equi_diametro<<endl;

    return res_eq_diametro;
  }


  float testaPerimetro(float p)
  {
    float res_perimetro = 0.0f;

    if (p > perimetro)
    {
      res_perimetro = ((100.0f * perimetro) / p);
    }
    else
    {
      res_perimetro = ((100.0f * p) / perimetro);
    }
    
    cout<<" res_perimetro "<<res_perimetro<<endl;
    res_perimetro /= 100.0f;
   
    return res_perimetro;
  }

  float testaHistograma(MatND h)
  {
    float res_histograma = compareHist(h, histograma, CV_COMP_CORREL);
    // cout<<" "<<res_histograma<<endl;
    return res_histograma;
  }
};

vector<Modelo> modelos;
Scalar cores[] = {
  Scalar(255, 0, 0),
  Scalar(255, 255, 0),
  Scalar(255, 0, 255),
  Scalar(0, 255, 0),
  Scalar(0, 0, 255),
  Scalar(0, 0, 128),
  Scalar(0, 128, 128),
  Scalar(128, 0, 255),
  Scalar(0,250, 255),
  Scalar(0, 255, 255)
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
    cvtColor(imagem(r), rhsv, CV_BGR2HSV);
    contorno = c;
    float area = contourArea(c);

    vector< Point > _hull;
    convexHull( Mat(c), _hull, false );
    float _hull_area = contourArea(_hull);

    double _area_perimetro = arcLength(c, true);

    double _sol = area/_hull_area;
    double _equi_diametro = sqrt(4*area/pi);

    int rhistsz = 180;    // bin size
    float range[] = { 0, 180};
    const float *ranges[] = { range };
    int channels[] = {0};

    calcHist( &rhsv, 1, channels, Mat(), rhist, 1, &rhistsz, ranges, true, false );
    normalize(rhist,rhist,0,255,NORM_MINMAX, -1, Mat() );

    wind = r;
    int _classe = -1;
    double maior = -1;

    for (int j = 0; j < (int)modelos.size(); j++) {
      
      /**
       * Obtêm a similaridade dos objetos
       */
      double _area = modelos[j].testaArea(rhist,area);
      double _hull = modelos[j].testaHullArea(_hull_area);
      double _histograma = modelos[j].testaHistograma(rhist);
      // double _perimetro = modelos[j].testaPerimetro(_area_perimetro);
      double _solidity = modelos[j].testaSolidity(_sol);
      double _diametro = modelos[j].testaDiamentroEquivalente(_equi_diametro);
      
      double similaridade = 0.1f*_histograma + 0.2f*_area + 0.3f*_hull + 0.2f*_solidity + 0.2f*_diametro;
      // cout<<"similaridade = "<<similaridade<<endl;
      
      if (similaridade > 0.8 && similaridade > maior) 
      {
        _classe = j;
        maior = similaridade;
        break;
      }
    }

    if (_classe == -1) 
    {
      Modelo m;
      m.area = area;
      m.histograma = rhist;
      m.hull_area = _hull_area;
      m.perimetro = _area_perimetro;
      m.solidity = _sol;
      m.equi_diametro = _equi_diametro;

      modelos.push_back(m);

      _classe = (int)modelos.size() - 1;
    }

    classe = _classe;
  }

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
 bool
  flag1 = false,
  flag2 = false,
  flag3 = false,
  flag4 = true;

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

MatND calculaHistograma(Mat &img)
{
  int rhistsz = 180;    // bin size
  float range[] = { 0, 180};
  const float *ranges[] = { range };
  int channels[] = {0};
  MatND rhist;
  Mat rhsv;
  cvtColor(img, rhsv, CV_BGR2HSV);

  calcHist( &rhsv, 1, channels, Mat(), rhist, 1, &rhistsz, ranges, true, false );

  return rhist;
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

  // pega X frames para estabilizar background
  for (int i = 0; i < 80; i++) {
    *cap >> frame;
    mog2->apply(frame, mascara_background);
  }

  img = frame.clone();
  namedWindow("video", WINDOW_AUTOSIZE);

  // matriz de morfologia
  Mat elemento = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1,1) );

  pegaROI();

  cvNamedWindow( "Binario", CV_WINDOW_AUTOSIZE );

  vector<int> contadores(10, 0);

  vector<Objeto> objetos;
  bool pausar = false;

  double hframe = frame.size().height;
  double wframe = frame.size().width;

  Point p1(0, roi.y);
  Point p2(wframe, roi.y);
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
      threshold(mascara_background, binaryImg, 50, 255, CV_THRESH_BINARY);

      Mat binOrig = binaryImg.clone();

      morphologyEx(binaryImg, binaryImg, CV_MOP_ERODE, elemento);
      morphologyEx(binaryImg, binaryImg, CV_MOP_ERODE, elemento);
      // for (int t = 0; t < 1; t++)
      // {
      //   morphologyEx(binaryImg, binaryImg, CV_MOP_OPEN, elemento);
      // }

      Mat ContourImg = binaryImg.clone();

      // pega contornos
      vector< vector<Point> > contornos;
      findContours(ContourImg, contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

      // Cria uma imagem preta, pintando os contornos de branco
      Mat mask = Mat::zeros(ContourImg.rows, ContourImg.cols, CV_8UC1);
      drawContours(mask, contornos, -1, Scalar(255, 255, 255), CV_FILLED);

      // cria uma nova imagem, sobrepondo o frame com a máscara
      // assim pinta somente os objetos, num fundo branco
      Mat crop(frame.rows, frame.cols, frame.type());
      crop.setTo(Scalar(0,0,0));
      frame.copyTo(crop, mask);


      for (int i = 0; i < (int)contornos.size(); i++) 
      {
        Rect bb = boundingRect(contornos[i]);

        // objeto muito pequeno, cai fora
        if (bb.width <= 5 || bb.height <= 5)
        {
          continue;
        }
        else if (bb.width >= 70 || bb.height >= 50)
        {
          continue;
        }

        // verifica se objeto já foi detectado em um frame anterior
        // por exemplo se a leitura e muito rapida
        bool achou = false;

        for (int j = 0; j < (int)objetos.size(); j++) {
          Rect intersecao = bb & objetos[j].wind;

          if (intersecao.height > 15 || intersecao.width > 9) {
            achou = true;
          }
        }

        if (achou) 
        {
          continue;
        }

        // se não encontrar, cria um novo objeto
        if (bb.y >= roi.y && bb.y <= roi.y + 25) {
          Objeto o(crop, bb, contornos[i]);
          objetos.push_back(o);
        }
      }

      // percorre objetos, chamando Camshift
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

        // se a janela do camshift ficar muito grande, remove objeto
        if (objetos[i].wind.height > 70 || objetos[i].wind.width > 70)
        {
          continue;
        }

        // se objeto deve continuar, coloca-o em um novo vetor
        novos_objetos.push_back(objetos[i]);
      }

      objetos = novos_objetos;

      int offset = 1;
      for (int i = 0; i < (int) contadores.size(); i++)
      {
        if (contadores[i] < 1) {
          continue;
        }

        // mostra contagem na tela
        string text = "Contador: " + to_string(contadores[i]);
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.7;
        int thickness = 2;
        cv::Point textOrg(10, 30 * offset);

        putText(frame, text, textOrg, fontFace, fontScale, cores[i], thickness,8);
        offset++;
      }

      line(frame, p1, p2, Scalar(0, 0, 255), 3);
      line(frame, p3, p4, Scalar(0, 0, 255), 3);

      imshow("video", frame);
      imshow("Binario", binaryImg);
    }

    int key =  waitKey(10);

    if (key == 27) 
    {
      break;
    }
    else if (key == 112)
    {
      pausar = !pausar;
    }
  }
  int total = 0;
  for (int i = 0; i < (int) contadores.size(); i++)
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

