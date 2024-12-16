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

class Colorize extends StatefulWidget {
  const Colorize({super.key});

  @override
  _ColorizeState createState() => _ColorizeState();
}

class _ColorizeState extends State<Colorize> {
  File? _image;
  String? _imageBase64;
  String? _colorizedImageBase64;
  bool _isLoading = false;

  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      if (kIsWeb) {
        final bytes = await pickedFile.readAsBytes();
        setState(() {
          _imageBase64 = base64Encode(bytes);
          _colorizedImageBase64 = null;
        });
      } else {
        setState(() {
          _image = File(pickedFile.path);
          _colorizedImageBase64 = null;
        });
      }
    }
  }

  Future<void> _colorizeImage() async {
    if (_image == null && _imageBase64 == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No image selected.')),
      );
      return;
    }
    
    setState(() {
      _isLoading = true;
      _colorizedImageBase64 = null;
    });
    
    try {
      final uri = Uri.parse('${Config.IPaddress}/colorize');
      var request = http.MultipartRequest('POST', uri);

      if (!kIsWeb) {
        request.files.add(await http.MultipartFile.fromPath(
          'image',
          _image!.path,
          contentType: MediaType('image', 'png'),
        ));
      } else {
        request.fields['image_base64'] = _imageBase64!;
      }

      var response = await request.send();

      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        final jsonResponse = json.decode(responseBody);

        setState(() {
          _colorizedImageBase64 = jsonResponse['colorizedImage'];
          _isLoading = false;
        });
      } else {
        setState(() {
          _isLoading = false;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to colorize image. Status: ${response.statusCode}')),
        );
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error occurred: $e')),
      );
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
          'SAR Image Colorization',
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
            return _buildWebHome(); 
          } else {
            return _buildMobileHome();
          }
        },
      ),
    );
  }

  // Mobile Home Layout
  Widget _buildMobileHome() {
    return SingleChildScrollView(
      child: ConstrainedBox(
        constraints: BoxConstraints(
          minHeight: MediaQuery.of(context).size.height - 50,
        ),
        child: IntrinsicHeight(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 40),
            width: double.infinity,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                const Column(
                  children: [
                    Text(
                      'SAR Image Colorization',
                      style: TextStyle(
                        fontSize: 30,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 20),
                    Text(
                      "Upload a SAR image to colorize.",
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
                    makeButton("Colorize", Icons.color_lens, _colorizeImage),
                  ],
                ),
                if (_isLoading)
                  const CircularProgressIndicator()
              ],
            ),
          ),
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
                "SAR Image Colorization",
                style: TextStyle(
                  fontSize: 35,
                  fontWeight: FontWeight.bold,
                  color: Color.fromARGB(255, 29, 81, 111),
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                "Upload a SAR image to colorize.",
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
                onPressed: _colorizeImage,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color.fromARGB(255, 29, 81, 111),
                  padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 100),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(50),
                  ),
                ),
                child: const Text(
                  "Colorize",
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

      return Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // First Column: Original Image
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      'Original Image',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                    const SizedBox(height: 10),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: Container(
                        decoration: BoxDecoration(
                          border: Border.all(color: Colors.grey.shade300),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: kIsWeb
                          ? Image.network(
                              'data:image/jpeg;base64,$_imageBase64',
                              height: 200,
                              fit: BoxFit.cover,
                            )
                          : Image.file(
                              _image!,
                              height: 200,
                              fit: BoxFit.cover,
                            ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(width: 20), // Spacing between columns

              // Second Column: Ground Truth Image
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      'Ground Truth',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                    const SizedBox(height: 10),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: Container(
                        decoration: BoxDecoration(
                          border: Border.all(color: Colors.grey.shade300),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: _image != null || _imageBase64 != null
                          ? Image.asset(
                              'assets/color_mask.jpeg',
                              height: 200,
                              fit: BoxFit.cover,
                            )
                          : const Text("No ground truth", style: TextStyle(color: Colors.grey)),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 20), // Spacing between rows
          
          // Third Row: Colorized Image
          Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Text(
                'Colorized Image',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
              const SizedBox(height: 10),
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey.shade300),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: _colorizedImageBase64 != null
                    ? Image.memory(
                        base64Decode(_colorizedImageBase64!),
                        height: 300, // Increased height to make it more prominent
                        fit: BoxFit.cover,
                      )
                    : const Text("Waiting to colorize", style: TextStyle(color: Colors.grey)),
                ),
              ),
            ],
          ),
        ],
      );
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