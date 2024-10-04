import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'main.dart'; // Import the main.dart file to navigate back to MyApp

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image; // Store the selected image
  String? _classificationResult; // Store the classification result
  bool _isLoading = false; // Loading state for when the image is being classified

  final ImagePicker _picker = ImagePicker();

  // Function to pick an image from the gallery
  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _classificationResult = null; // Clear previous result
      });
    }
  }

  // Function to upload the image to the backend and get classification
  Future<void> _classifyImage() async {
    if (_image == null) {
      setState(() {
        _classificationResult = 'No image selected.';
      });
      return;
    }
    setState(() {
      _isLoading = true;
    });
    try {
      final uri = Uri.parse('http://192.168.1.140:5000/classify');
      var request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath(
          'image', 
          _image!.path,
          contentType: MediaType('image', 'jpeg'),
        ));

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

  // Function to log out the user and navigate to the main screen
  Future<void> _logout() async {
    await FirebaseAuth.instance.signOut(); // Sign out from Firebase
    // Navigate back to the root (MyApp), which handles auth check
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(builder: (context) => const MyApp()), // Go to MyApp
      (Route<dynamic> route) => false, // Remove all previous routes
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
            onPressed: _logout, // Call the logout function
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 40),
          height: MediaQuery.of(context).size.height - 50,
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
                  _image == null
                      ? const Text("No image selected.")
                      : ClipRRect(
                          borderRadius: BorderRadius.circular(10),
                          child: Container(
                            decoration: BoxDecoration(
                              border: Border.all(color: Colors.grey.shade300),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Image.file(_image!, height: 200, fit: BoxFit.cover),
                          ),
                        ),
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
      ),
    );
  }

  // Create a button with consistent styling
  Widget makeButton(String label, IconData icon, VoidCallback onPressed) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 5),
      child: Material(
        elevation: 5, // Add elevation for shadow effect
        borderRadius: BorderRadius.circular(50),
        child: MaterialButton(
          minWidth: double.infinity,
          height: 60,
          onPressed: onPressed,
          color: const Color.fromARGB(255, 29, 81, 111),
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(50),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, color: Colors.white),
              const SizedBox(width: 10),
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w800,
                  fontSize: 18,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
