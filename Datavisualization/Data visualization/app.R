
#Install packages if missing
list.of.packages <- c("ggplot2", "grid", "gridExtra","dbplyr", "ltm", "data.table", "readxl", "ggpmisc", "ggpubr", "DT", "shinyWidgets", "knitr","plotly","tidyverse","shiny", "htmlTable", "jtools", "tableHTML")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


library(shiny)
library(tidyverse)
library(htmlTable)
library(knitr)
library(jtools)
library(tableHTML)
library(readxl)

# load 
collected_data1 <- read_excel('data/for data vis.xlsx')
#Load another data
collected_data <- read_excel('data/for data vis.xlsx', sheet = "consolidated")

#Read the seasonal information data
dfx <- read_excel('data/que2.xlsx')


#Convert month to year
collected_data$Year <- factor(collected_data$Year)
#Convert month to year
collected_data1$Month <- factor(collected_data1$Month)


# Pre-compute some variables to be used by app
not_numeric <- sapply(names(collected_data), function(x) !is.numeric(collected_data[[x]]))

data <- collected_data
#elect numeric variables
nums <- unlist(lapply(collected_data, is.numeric))

data_num <- collected_data[,nums]
#Summarize for mean 
dfd <- collected_data1 %>% group_by(Year) %>% summarise("Average Sales Amount" = mean(Amount))


#Summarize for mean 
dfd1 <- collected_data %>% group_by(Year) %>% summarise("Total Sales Amount"= sum(TOTAL))

#Summarize for mean 
dfd2 <- collected_data %>% group_by(Category) %>% summarise("Total Sales Amount" = sum(TOTAL))

#Summarize for mean 
dfd3 <- collected_data %>% group_by(Category) %>% summarise("Total Sales Amount" = sum(TOTAL))


#Further summaries for the seasonal data
#Summarize for mean 
dfx1 <- dfx %>% group_by(Year) %>% summarise("Average Sales Amount"= sum(Sales))

#Summarize for mean 
dfx2 <- dfx %>% group_by(Season) %>% summarise("Total Sales Amount" = sum(Sales))

#Summarize for mean 
dfxx2 <- dfx %>% group_by(Season) %>% summarise("Average Sales Amount" = mean(Sales))

#Summarize for mean 
dfx3 <- dfx %>% group_by(Category) %>% summarise("Total Sales Amount" = sum(Sales))

namesx <- c("Jan" ,     "Feb" ,     "Mar",      "Apr",      "May",      "Jun",      "Jul" ,     "Aug" ,     "Sep" ,     "Oct" ,     "Nov"  ,    "Dec"  ,    "TOTAL")
# Define UI ----
ui <- fluidPage(
  
  titlePanel("Sales Data Analysis"),
  
  p("A Shiny APP suitable for exploratory data analysis, correlation, and regression analyses. The output in the
    'Summary' tab is based on the full dataset. The 
    controls in the sidebar are for the 'Plot','Data Overview', 'Analysis of seasonal information', and 'Analysis of total sales data'. The
    'Data Overview' tab shows a maximum of 15 rows of the dataset. The correlation shows the correlation test between the dependent and independent Variables"),
  
  
  
  sidebarPanel(
    
    sliderInput("sampleSize", "Plot sample size (n)", min = 1, max = nrow(data),
                value = min(1000, nrow(data)), step = nrow(data) / 50, round = 0),
    radioButtons("smpType", "Plot sample type",
                 choices = list("Random k observations" = "random", "First k observations" = "first")),
    numericInput("smpseed", "Random Sample Seed", value = 10),
    
    selectInput("x", "X", names(data)),
    selectInput("y", "Y", c("None", namesx), namesx[[2]]),
    
    # only allow non-numeric variables for color
    selectInput("color", "Color (Filter by level)", c("None", names(data)[not_numeric])),
    
    p("Jitter and smoothing are used for numeric variables only
      are selected."),
    checkboxInput("jitter", "Jitter"),
    checkboxInput("smooth", "Smooth")
  ),
  
  mainPanel(
    
    # Output: Tabset
    tabsetPanel(type = "tabs",
                tabPanel("Interactive Plots", plotOutput("plot")),
                tabPanel("Data Overview", verbatimTextOutput("Overview")),
                tabPanel("Summary", verbatimTextOutput("summary")),
                tabPanel("Static graphs for data with seasonal information", fluidRow(
                  column(6,plotOutput('plot3'),column(12,plotOutput('total_sales1')),fluidRow(
                    column(12,verbatimTextOutput("sales_category1")),
                    column(12,verbatimTextOutput('sales1'))
                  ))
                )),
                tabPanel("Static graphs for total sales data", fluidRow(
                                                        column(6,plotOutput('plot1'),column(12,plotOutput('total_sales')),fluidRow(
                                                                                                      column(12,verbatimTextOutput("sales_category")),
                                                                                                      column(12,verbatimTextOutput('sales'))
                                                        ))
                ))
    )
  )
)

# Define server logic ----
server <- function(input, output) {
  pl <- ggplot(data=dfd, aes(x=Year, y=`Average Sales Amount`)) +
    geom_line(color = "steelblue")+ggtitle("Distribution of average sales")+ylab("Total Sales Amount")+
    theme(axis.text.x = element_text(angle = 60, vjust = 0.5, hjust=0.5), plot.title = element_text(size = rel(1.8), face = "bold", hjust = 0.5),
          axis.title = element_text(size = rel(1.5)))

  p22 <- ggplot(data=dfd1, aes(x=Year, y=`Total Sales Amount`)) +
      geom_bar(color = "steelblue",stat = "identity", fill = "steelblue")+ggtitle("Distribution of Total sales")+ylab("Total Sales Amount")+
    theme(axis.text.x = element_text(angle = 60, vjust = 0.5, hjust=0.5), plot.title = element_text(size = rel(1.8), face = "bold", hjust = 0.5),
          axis.title = element_text(size = rel(1.5)))
    
  
  #Sales information data
  pl1 <- ggplot(data=dfx1, aes(x=Year, y=`Average Sales Amount`)) +
    geom_line(color = "steelblue")+ggtitle("Distribution of Total sales")+ylab("Total Sales Amount")+
    theme(axis.text.x = element_text(angle = 60, vjust = 0.5, hjust=0.5), plot.title = element_text(size = rel(1.8), face = "bold", hjust = 0.5),
          axis.title = element_text(size = rel(1.5)))
  
  p222 <- ggplot(data=dfxx2, aes(x=Season, y=`Average Sales Amount`)) +
    geom_bar(color = "steelblue",stat = "identity", fill = "steelblue")+ggtitle("Distribution of Average Sales per Season")+ylab("Total Sales Amount")+
    theme(axis.text.x = element_text(angle = 60, vjust = 0.5, hjust=0.5), plot.title = element_text(size = rel(1.8), face = "bold", hjust = 0.5),
          axis.title = element_text(size = rel(1.5)))
  
  
  
  # Generate data summaries
  output$summary <- renderPrint({
    kable(summary(collected_data), caption = "Summary Statistics")
  })
  
  
  # Generate summary
  output$sales<- renderPrint({
    kable(dfd, caption = "Average sales per year")
  })
  

  
  # Generate data summaries
  output$sales_category1<- renderPrint({
    kable(dfd1, caption = "Total sales per year")
  })
  
  
  # Generate summary
  output$sales1<- renderPrint({
    kable(dfx1, caption = "Total sales per year")
  })
  
  
  
  # Generate data summaries
  output$sales_category1<- renderPrint({
    kable(dfx2, caption = "Average sales per Season")
  })
  
  

  # get new dataset sample for plotting
  idx <- reactive({
    if (input$smpType == "first") {
      1:input$sampleSize
    } else {
      set.seed(input$smpseed)
      sample(nrow(collected_data), input$sampleSize)
    }
  })
  data <- reactive(collected_data[idx(), , drop = FALSE])
  
  # Get head of selected data
  output$Overview <- renderPrint({
    kable(head(data(), n = 15), caption = "First 15 rows of the data")
  })
  
  
  # get plot type
  # * 2: both numeric variables
  # * 1: one numeric, one non-numeric variable
  # * 0: both non-numeric variables
  # * -1: only one variable provided
  plot_type <- reactive({
    if (input$y != "None")
      is.numeric(collected_data[[input$x]]) + is.numeric(collected_data[[input$y]])
    else
      -1
  })
  
  

  output$plot1 = renderPlot({pl})
  # Generate plot
  output$total_sales = renderPlot({p22})
  
  
  output$plot3 = renderPlot({pl1})
  # Generate plot
  output$total_sales1 = renderPlot({p222})
  
  
  # Create plot
  output$plot <- renderPlot({
    if (plot_type() == 2) {
      # both numeric variables: scatterplot
      # also allow for color, jitter & smoothing
      pltt <- ggplot(data(), aes_string(x = input$x, y = input$y), color = "steelblue")
      pltt <- pltt + geom_smooth(method='lm')
      
      if (input$jitter)
        pltt <- pltt + geom_jitter(alpha = 0.5)
      else
        pltt <- pltt + geom_point(alpha = 0.5)
      
      if (input$smooth)
        pltt <- pltt + geom_smooth()
      
      # color change
      if (input$color != "None")
        pltt <- pltt + aes_string(color = input$color)
    } else if (plot_type() == 1) {
      # one numeric var, one character var: boxplot
      # allow color, don't allow jitter or smoothing
      pltt <- pltt <- ggplot(data(), aes_string(x = input$x, y = input$y)) + 
        geom_boxplot()
      
      # fill change
      if (input$color != "None")
        pltt <- pltt + aes_string(fill = input$color)
    } else if (plot_type() == 0) {
      # two character variables: heatmap
      # don't allow color, jitter or smoothing
      datafx <- reactive(data()[, c(input$x, input$y), drop = FALSE] %>%
                            group_by(across()) %>%
                            summarize(count = n())
      )
      pltt <- ggplot(datafx(), 
                  mapping = aes_string(x = input$x, y = input$y, fill = "count")) +
        geom_tile() +
        scale_fill_gradient(low = "#e7e7fd", high = "#1111dd")
    } else {
      # only one variable: univariate plot
      # allow color, don't allow jitter or smoothing
      pltt <- ggplot(data(), aes_string(x = input$x))
      
      if (is.numeric(collected_data[[input$x]]))
        pltt <- pltt + geom_histogram()
      else
        pltt <- pltt + geom_bar()
      
      # fill change
      if (input$color != "None")
        pltt <- pltt + aes_string(fill = input$color)
    }
    
    # add title
    if (plot_type() >= 0) {
      pltt <- pltt + labs(title = paste(input$y, "vs.", input$x))
    } else {
      pltt <- pltt + labs(title = paste("Distribution of", input$x))
    }
    
    # add styling
    pltt <- pltt + 
      theme_bw() +
      theme(axis.text.x = element_text(angle = 60, vjust = 0.5, hjust=0.5), plot.title = element_text(size = rel(1.8), face = "bold", hjust = 0.5),
            axis.title = element_text(size = rel(1.5)))
    
    print(pltt)
    
  }, height=600)
}

# Run the app ----
shinyApp(ui = ui, server = server)