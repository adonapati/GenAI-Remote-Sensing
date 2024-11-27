import 'package:flutter/material.dart';
import 'package:remote_sensing/FloodDetection.dart';
import 'CropClassification.dart'; // Import the CropClassification page

class SelectionPage extends StatefulWidget {
  const SelectionPage({super.key});

  @override
  _SelectionPageState createState() => _SelectionPageState();
}

class _SelectionPageState extends State<SelectionPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      backgroundColor: Colors.white,
      appBar: AppBar(
        elevation: 0,
        backgroundColor: Colors.white,
        title: const Text(
          'Select Service',
          style: TextStyle(color: Colors.black),
        ),
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          if (constraints.maxWidth > 600) {
            return _buildWebLayout();
          } else {
            return _buildMobileLayout();
          }
        },
      ),
    );
  }

  // Mobile Layout
  Widget _buildMobileLayout() {
    return SingleChildScrollView(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 40),
        height: MediaQuery.of(context).size.height - 50,
        width: double.infinity,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Column(
              children: [
                Text(
                  'SAR',
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 20),
                Text(
                  "Choose a service to get started",
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 15,
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
            const SizedBox(height: 50),
            _buildServiceButton(
              "Crop Classification\n(Deep Learning)",
              Icons.grass,
              () => Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const CropClassification()),
              ),
            ),
            const SizedBox(height: 30),
            _buildServiceButton(
              "Flood Detection\n(Generative AI)",
              Icons.water_damage,
                () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const FloodDetection()),
                ),
            ),
          ],
        ),
      ),
    );
  }

  // Web Layout
  Widget _buildWebLayout() {
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
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "Agricultural AI Services",
                style: TextStyle(
                  fontSize: 35,
                  fontWeight: FontWeight.bold,
                  color: Color.fromARGB(255, 29, 81, 111),
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                "Choose a service to get started",
                style: TextStyle(
                  color: Colors.grey,
                  fontSize: 16,
                  fontWeight: FontWeight.w400,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 50),
              _buildServiceButton(
                "Crop Classification\n(Deep Learning)",
                Icons.grass,
                () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const CropClassification()),
                ),
              ),
              const SizedBox(height: 30),
              _buildServiceButton(
                "Flood Detection\n(Generative AI)",
                Icons.water_damage,
                () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const FloodDetection()),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // Service Button Widget
  Widget _buildServiceButton(String label, IconData icon, VoidCallback onPressed) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: const Color.fromARGB(255, 29, 81, 111),
        padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 50),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(50),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: Colors.white, size: 30),
          const SizedBox(width: 10),
          Text(
            label,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w800,
              color: Colors.white,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}