name := """shapelets_api_2"""

version := "1.0"

scalaVersion := "2.12.1"

// Change this to another test framework if you prefer
libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "latest.release",
  "org.scalanlp" %% "breeze-natives" % "latest.release",
  "org.scalanlp" %% "breeze-viz" % "latest.release"
    )
