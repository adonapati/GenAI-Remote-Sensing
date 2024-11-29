import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'dart:convert'; // For base64 encoding
import 'main.dart';
import 'config.dart';

class CropClassification extends StatefulWidget {
  const CropClassification({super.key});

  @override
  _CropClassificationState createState() => _CropClassificationState();
}

class _CropClassificationState extends State<CropClassification> {
  File? _image;
  String? _imageBase64;
  String? _classificationResult;
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
        });
      } else {
        setState(() {
          _image = File(pickedFile.path);
          _classificationResult = null;
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
    });
    try {
      final uri = Uri.parse('${Config.IPaddress}/classify_crop');
      var request = http.MultipartRequest('POST', uri);

      if (!kIsWeb) {
        request.files.add(await http.MultipartFile.fromPath(
          'image',
          _image!.path,
          contentType: MediaType('image', 'jpeg'),
        ));
      } else {
        request.fields['image_base64'] = _imageBase64!;
      }

      var response = await request.send();

      if (response.statusCode == 200) {
        final responseData = await http.Response.fromStream(response);
        setState(() {
          _classificationResult = responseData.body;
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

  Future<void> _logout() async {
    await FirebaseAuth.instance.signOut();
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (context) => const MyApp()),
      (Route<dynamic> route) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      backgroundColor: Colors.white,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.white,
        title: const Text(
          'Home',
          style: TextStyle(color: Colors.black),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout, color: Colors.black),
            onPressed: _logout,
          ),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          if (constraints.maxWidth > 600) {
            return _buildWebHome(); // Use the fancy web layout
          } else {
            return _buildMobileHome(); // Keep the mobile layout
          }
        },
      ),
    );
  }

  // Mobile Home Layout
  Widget _buildMobileHome() {
    return SingleChildScrollView(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 40),
        height: MediaQuery.of(context).size.height - 50,
        width: double.infinity,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Column(
              children: [
                const Text(
                  'Home Page',
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 20),
                const Text(
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
            if (_isLoading)
              const CircularProgressIndicator()
            else if (_classificationResult != null)
              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: Text(
                  "Classification Result: $_classificationResult",
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  // Web Home Layout
Widget _buildWebHome() {
  return Center(
    child: Container(
      width: 500,
      padding: const EdgeInsets.all(40),
      margin: const EdgeInsets.symmetric(vertical: 50),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 15,
            spreadRadius: 5,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: SingleChildScrollView( // Add SingleChildScrollView here
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
            if (_isLoading)
              const Padding(
                padding: EdgeInsets.only(top: 20),
                child: CircularProgressIndicator(),
              )
            else if (_classificationResult != null)
              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: Text(
                  "Classification Result: $_classificationResult",
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
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
