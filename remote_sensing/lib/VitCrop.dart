import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'dart:convert'; // For base64 encoding
import 'config.dart';

class VitCrop extends StatefulWidget {
  const VitCrop({super.key});

  @override
  _VitCropState createState() => _VitCropState();
}

class _VitCropState extends State<VitCrop> {
  File? _image;
  String? _imageBase64;
  String? _classificationResult;
  double? _confidence;
  bool _isLoading = false;

  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      if (kIsWeb) {
        final bytes = await pickedFile.readAsBytes();
        setState(() {
          _imageBase64 = base64Encode(bytes);
          _classificationResult = null;
          _confidence = null;
        });
      } else {
        setState(() {
          _image = File(pickedFile.path);
          _classificationResult = null;
          _confidence = null;
        });
      }
    }
  }

  Future<void> _classifyImage() async {
    if (_image == null && _imageBase64 == null) {
      setState(() {
        _classificationResult = 'No image selected.';
      });
      return;
    }
    
    setState(() {
      _isLoading = true;
      _classificationResult = null;
      _confidence = null;
    });

    try {
      final uri = Uri.parse('${Config.IPaddress}/vit_classification');
      var request = http.MultipartRequest('POST', uri);

      if (!kIsWeb && _image != null) {
        request.files.add(await http.MultipartFile.fromPath(
          'image',
          _image!.path,
          contentType: MediaType('image', 'jpeg'),
        ));
      } else if (kIsWeb && _imageBase64 != null) {
        request.fields['image_base64'] = _imageBase64!;
      }

      var response = await request.send();

      if (response.statusCode == 200) {
        final responseData = await http.Response.fromStream(response);
        final jsonResponse = jsonDecode(responseData.body);
        
        setState(() {
          _classificationResult = jsonResponse['predictions'];
          _confidence = jsonResponse['confidence'];
          _isLoading = false;
        });
      } else {
        setState(() {
          _classificationResult = 'Failed to predict image. Status: ${response.statusCode}';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _classificationResult = 'Error occurred: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          if (constraints.maxWidth > 600) {
            return _buildWebHome();
          } else {
            return _buildMobileHome();
          }
        },
      ),
    );
  }

  // Modify the result display in both mobile and web layouts
  Widget _buildResultWidget() {
    if (_isLoading) {
      return const CircularProgressIndicator();
    } else if (_classificationResult != null) {
      return Column(
        children: [
          Text(
            "Classification Result: $_classificationResult",
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          if (_confidence != null)
            Text(
              "Confidence: ${(_confidence! * 100).toStringAsFixed(2)}%",
              style: const TextStyle(
                fontSize: 14,
                color: Colors.grey,
              ),
            ),
        ],
      );
    } else {
      return const SizedBox.shrink();
    }
  }

  // Update mobile home layout
  Widget _buildMobileHome() {
    return SingleChildScrollView(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 40),
        height: MediaQuery.of(context).size.height - 50,
        width: double.infinity,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            const Column(
              children: [
                Text(
                  'Home Page',
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 20),
                Text(
                  "Upload an image to classify it.",
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 15,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
            Column(
              children: [
                _buildImagePreview(),
                const SizedBox(height: 20),
                makeButton("Select Image", Icons.image, _pickImage),
                const SizedBox(height: 20),
                makeButton("Classify Image", Icons.cloud_upload, _classifyImage),
              ],
            ),
            Padding(
              padding: const EdgeInsets.only(top: 20),
              child: _buildResultWidget(),
            ),
          ],
        ),
      ),
    );
  }

  // Update web home layout
  Widget _buildWebHome() {
    return Center(
      child: Container(
        width: 500,
        padding: const EdgeInsets.all(40),
        margin: const EdgeInsets.symmetric(vertical: 50),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          boxShadow: const [
            BoxShadow(
              color: Colors.black12,
              blurRadius: 15,
              spreadRadius: 5,
              offset: Offset(0, 10),
            ),
          ],
        ),
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "Home Page",
                style: TextStyle(
                  fontSize: 35,
                  fontWeight: FontWeight.bold,
                  color: Color.fromARGB(255, 29, 81, 111),
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                "Upload an image to classify it.",
                style: TextStyle(
                  color: Colors.grey,
                  fontSize: 16,
                  fontWeight: FontWeight.w400,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 30),
              _buildImagePreview(),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _pickImage,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color.fromARGB(255, 29, 81, 111),
                  padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 100),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(50),
                  ),
                ),
                child: const Text(
                  "Select Image",
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _classifyImage,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color.fromARGB(255, 29, 81, 111),
                  padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 100),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(50),
                  ),
                ),
                child: const Text(
                  "Classify Image",
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
              ),
              const SizedBox(height: 20),
              _buildResultWidget(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImagePreview() {
    if (_image == null && _imageBase64 == null) {
      return const Text("No image selected.");
    }

    if (kIsWeb) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(10),
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: Colors.grey.shade300),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Image.network(
            'data:image/jpeg;base64,$_imageBase64',
            height: 200,
            fit: BoxFit.cover,
          ),
        ),
      );
    } else {
      return ClipRRect(
        borderRadius: BorderRadius.circular(10),
        child: Container(
          decoration: BoxDecoration(
            border: Border.all(color: Colors.grey.shade300),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Image.file(
            _image!,
            height: 200,
            fit: BoxFit.cover,
          ),
        ),
      );
    }
  }

  Widget makeButton(String label, IconData icon, VoidCallback onPressed) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, color: Colors.white),
      label: Text(label, style: const TextStyle(fontSize: 18)),
      style: ElevatedButton.styleFrom(
        backgroundColor: const Color.fromARGB(255, 29, 81, 111),
        padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 30),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
      ),
    );
  }
}
